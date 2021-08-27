import copy
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
from dataclasses import dataclass
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, PreTrainedTokenizer, T5Config
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils import logging

from src.data.formatting import add_target_prefix

logger = logging.get_logger(__name__)
# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
_HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@dataclass
class VariationalT5EncoderOutput(ModelOutput):
    """
    Same attributes as the BaseModelOutputWithPastAndCrossAttentions class (output of the regular
    encoder, T5Stack)

    And some attributes from the VAE encoder:
        q_phi
        vae_latent
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # variational posterior of the VAE encoder
    q_phi: Normal = None
    # the latent z sampled from q_phi after encoding
    vae_latent: torch.Tensor = None


@dataclass
class GT8ModelOutput(ModelOutput):
    """
    Same attributes as the BaseModelOutputWithPastAndCrossAttentions class (output of the regular
    encoder, T5Stack)

    And some attributes from the VAE model:
        recon_loss: reconstruction loss (identical to loss, if we don't use VAE)
        reg_loss: either KL(q(z|x)||p(z)) or MMD(q(z)||p(z))
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vae_latent: Optional[torch.FloatTensor] = None  # used by CycleVAE with dual loss
    recon_loss: Optional[torch.FloatTensor] = None
    reg_loss: Optional[torch.FloatTensor] = None


class VariationalT5Encoder(T5Stack):
    def __init__(self, config: T5Config, embed_tokens: nn.Embedding = None):
        super().__init__(config, embed_tokens)

        # additional layers for the variational posterior
        model_dim = config.d_model
        self.mu_z = nn.Linear(model_dim, model_dim)
        self.log_sigma_z = nn.Linear(model_dim, model_dim)

    def forward(self, *args, **kwargs):
        # make sure the output is always BaseModelOutputWithPastAndCrossAttentions
        assert kwargs["return_dict"]
        # call T5Stack.forward (using forward, not __call__, see
        # https://discuss.pytorch.org/t/recursionerror-calling-super-call-in-forward/57387)
        encoder_outputs = super().forward(*args, **kwargs)
        mu_z = self.mu_z(encoder_outputs.last_hidden_state)
        log_sigma_z = self.log_sigma_z(encoder_outputs.last_hidden_state)

        q_phi = Normal(loc=mu_z, scale=torch.exp(log_sigma_z))
        vae_z = q_phi.rsample()  # same dimension as mu_z (N, T, model_dim)

        return VariationalT5EncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            q_phi=q_phi,
            vae_latent=vae_z,
        )


class AddedStyleVAET5Encoder(T5Stack):
    def __init__(self, config: T5Config, embed_tokens: nn.Embedding = None):
        super().__init__(config, embed_tokens)

        model_dim = config.d_model
        # additional layers for the variational posterior q(s_x|x)
        self.mu_s_x = nn.Linear(model_dim, model_dim)
        self.log_sigma_s_x = nn.Linear(model_dim, model_dim)
        # and for q(s_y|y)
        self.mu_s_y = nn.Linear(model_dim, model_dim)
        self.log_sigma_s_y = nn.Linear(model_dim, model_dim)

    def forward(self, *args, **kwargs):
        # get input format (and raise an error if we are missing this argument)
        source_format = kwargs.pop("source")
        # make sure the output is always BaseModelOutputWithPastAndCrossAttentions
        assert kwargs["return_dict"]
        # call T5Stack.forward (using forward, not __call__, see
        # https://discuss.pytorch.org/t/recursionerror-calling-super-call-in-forward/57387)
        encoder_outputs = super().forward(*args, **kwargs)
        mean_sequence = encoder_outputs.last_hidden_state.mean(
            1
        )  # (N,T,dim) -> (N,dim)
        if source_format == "graph":
            mu_s = self.mu_s_x(mean_sequence)
            log_sigma_s = self.log_sigma_s_x(mean_sequence)

        elif source_format == "text":
            mu_s = self.mu_s_y(mean_sequence)
            log_sigma_s = self.log_sigma_s_y(mean_sequence)
        else:
            raise ValueError

        q_phi = Normal(loc=mu_s, scale=torch.exp(log_sigma_s))
        vae_s = q_phi.rsample()  # dimension (N, dim)

        return VariationalT5EncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            q_phi=q_phi,
            vae_latent=vae_s,
        )


class StyleVAET5Encoder(T5Stack):
    def __init__(self, config: T5Config, embed_tokens: nn.Embedding = None):
        super().__init__(config, embed_tokens)

        # additional layers for the variational posterior of x, q(s_x|x)
        model_dim = config.d_model
        self.mu_s_x = nn.Linear(model_dim, model_dim)
        self.log_sigma_s_x = nn.Linear(model_dim, model_dim)
        # and for q(s_y|y)
        self.mu_s_y = nn.Linear(model_dim, model_dim)
        self.log_sigma_s_y = nn.Linear(model_dim, model_dim)

    def forward(self, *args, **kwargs):
        # get input format (and raise an error if we are missing this argument)
        source_format = kwargs.pop("source")
        # make sure the output is always BaseModelOutputWithPastAndCrossAttentions
        assert kwargs["return_dict"]

        encoder_outputs = super().forward(*args, **kwargs)  # (N, T, dim)

        # representation of our special [STYLE] token, after encoding
        style_hidden_state = encoder_outputs.last_hidden_state[:, 0, :].clone()
        if source_format == "text":
            mu_s = self.mu_s_x(style_hidden_state)
            log_sigma_s = self.log_sigma_s_x(style_hidden_state)
        elif source_format == "graph":
            mu_s = self.mu_s_y(style_hidden_state)
            log_sigma_s = self.log_sigma_s_y(style_hidden_state)
        else:
            raise ValueError

        # variational posterior, and latent variable
        q_phi = Normal(loc=mu_s, scale=torch.exp(log_sigma_s))
        vae_s = q_phi.rsample()  # same dimension as mu (N, T, model_dim)

        return VariationalT5EncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            q_phi=q_phi,
            vae_latent=vae_s,
        )


class VAEBase(ABC):
    beta: float  # to be specified by Trainer, before calling model forward
    reg_loss_type: str  # to be specified during init

    def compute_reg_loss(self, q_phi: Normal, z: torch.Tensor):
        # N(0,I) prior: same shape (N, T, dim_z) and device than q_phi
        prior = Normal(
            loc=torch.zeros_like(q_phi.loc),
            scale=torch.ones_like(q_phi.scale),
        )

        if self.reg_loss_type == "kl":
            # (N, T, dim_z) as well, since kl_divergence on Normal distributions does not reduce result
            kl_div = kl_divergence(q_phi, prior)
            # reduce to a scalar by:
            #   - summing over latent dim, to obtain the D_KL between multivariate Normal distributions
            #   - taking the mean over batch and sequence dim, to match the
            #       CrossEntropyLoss (which takes mean over N and T as well)
            kl_div = kl_div.sum(dim=2).mean()
            return kl_div
        elif self.reg_loss_type == "mmd":
            # https://github.com/amir-abdi/disentanglement-pytorch/blob/master/models/infovae.py
            z_prior = prior.rsample()
            mmd = self.compute_mmd(z, z_prior)
            return mmd
        else:
            raise ValueError

    # MMD-VAE specific methods
    @staticmethod
    @abstractmethod
    def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        pass

    def compute_mmd(self, x: torch.Tensor, y: torch.Tensor):
        # original implementation: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        # MMD = E[k(x,x)] + E[k(y,y)] - 2 * E[k(x,y)]
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    # other methods, assuming latent variable is stored in encoder_outputs as vae_z
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        if "vae_latent" in kwargs:
            # use vae_z specified as a kwarg to the generate method
            encoder_outputs.vae_latent = kwargs["vae_latent"]
        else:
            # take mu_z (but we could also sample)
            encoder_outputs.vae_latent = encoder_outputs.q_phi.loc

        inputs = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: VariationalT5EncoderOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Expand tensors across the batch dimension, e.g. to have num_beams*batch_size
        instead of batch_size. Called during beam search generation for instance.

        We redefine this function to also expand vae_mu_z/vae_sigma_z (in encoder_outputs).
        """
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )

            # VAE specific code
            encoder_outputs["vae_z"] = encoder_outputs.vae_latent.index_select(
                0, expanded_return_idx.to(encoder_outputs.vae_latent.device)
            )
            loc = encoder_outputs.q_phi.loc.index_select(
                0, expanded_return_idx.to(encoder_outputs.q_phi.loc.device)
            )
            scale = encoder_outputs.q_phi.scale.index_select(
                0, expanded_return_idx.to(encoder_outputs.q_phi.scale.device)
            )
            encoder_outputs["q_phi"] = Normal(loc, scale)

            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs


class GT8Base(T5PreTrainedModel, ABC):
    # Based on T5ForConditionalGeneration, from transformers 4.9.1
    # https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    # Encoder class, to define (e.g. T5Stack for regular T5)
    encoder_cls = None

    def __init__(
        self,
        config,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_graph_token_id: int,
    ):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = self.encoder_cls(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.specify_target_with_prefix = specify_target_with_prefix
        if not specify_target_with_prefix:
            self.generate_text_token_id = generate_text_token_id
            self.generate_graph_token_id = generate_graph_token_id

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target=None,
        source=None,
    ):
        """
        target: Target format (can be "graph" or "text"). Only used when we don't specify it
        already with an added prefix in the inputs, otherwise it can be None.
        source: Source format ("graph" or "text"), passed to the encoder if it's not None.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # make things easier to read by using ModelOutputs objects for encoder/decoder outputs
        # -> just make sure this is never overridden to False
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        assert return_dict
        # same for model_parallel (and make sure we don't use it)
        assert not self.model_parallel
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training). Note: even in training, in some cases we might
        # want to re-use the same encoder_outputs (CycleVAE dual loss)
        if encoder_outputs is None:
            other_encoder_kwargs = {}
            if source is not None:
                other_encoder_kwargs["source"] = source
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **other_encoder_kwargs,
            )

        # to be fed to the decoder
        hidden_states = self.get_hidden_states(encoder_outputs)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(input_ids=labels, target=target)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss, recon_loss, reg_loss = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            recon_loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )

            loss, reg_loss = self.get_total_loss(
                recon_loss=recon_loss, encoder_outputs=encoder_outputs
            )

        return GT8ModelOutput(
            loss=loss,
            reg_loss=reg_loss,
            recon_loss=recon_loss,
            vae_latent=encoder_outputs.vae_latent
            if "vae_latent" in encoder_outputs
            else None,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past

    def _shift_right(self, input_ids, target):
        """
        Override the _shift_right method of T5PreTrainedModel, to add a
        custom token at the beginning of input_ids, (if self.specify_target_with_prefix).

        Instead of the default decoder_start_token_id (configured to pad_token_id for T5
        by default), use either a text_decoder_start_token_id or a graph_decoder_start_token_id
        depending on the desired target (graph or text generation).

        This is used during the forward, in training mode, to shift the labels and obtain decoder inputs
        During eval, we are using the base generate method, which takes care of using
        the right decoder_input_ids with the appropriate decoder_start_token_id argument.

        Args:
            input_ids:
            target: 'text' or 'graph'

        Returns:

        """
        pad_token_id = self.config.pad_token_id
        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."

        if self.specify_target_with_prefix:
            # target is already specified as a prefix in the input sequence
            decoder_start_token_id = pad_token_id
        elif target == "text":
            decoder_start_token_id = self.generate_text_token_id
        elif target == "graph":
            decoder_start_token_id = self.generate_graph_token_id
        else:
            raise ValueError(f"Target (text/graph) should be specified")
        assert (
            decoder_start_token_id is not None
        ), f"decoder_start_token_id (for target={target}) has not been defined)"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def generate_with_target(
        self,
        input_ids: torch.Tensor,
        target: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        method: str,
        num_beams=-1,
        **other_kwargs,
    ):
        """
        Call `generate` on our model, specifying the target format (graph/text)

        Args:
            input_ids:
            target: 'text' or 'graph'
            tokenizer:
            max_seq_length:
            method: 'greedy', 'beam_search', 'sample' or 'top_k'
            num_beams: Used only when method='beam_search'
            other_kwargs: 'vae_z' or 'source' format for instance. Will be passed to the model and the encoder

        Returns:

        """
        # specify the target format to the model
        if self.specify_target_with_prefix:
            input_ids = add_target_prefix(
                input_ids=input_ids,
                target=target,
                tokenizer=tokenizer,
                max_seq_len=max_seq_length,
            )
            decoder_start_token_id = None
        else:
            # don't touch the encoder_input_ids, but tell the decoder the target format
            if target == "text":
                decoder_start_token_id = self.generate_text_token_id
            elif target == "graph":
                decoder_start_token_id = self.generate_graph_token_id
            else:
                raise ValueError

        kwargs = {
            "input_ids": input_ids,
            "max_length": max_seq_length,
            "decoder_start_token_id": decoder_start_token_id,
        }
        kwargs.update(other_kwargs)

        # generate text according to the specified decoding method
        if method == "greedy":
            # nothing to change to config
            pass
        elif method == "beam_search":
            assert num_beams > 1
            kwargs["num_beams"] = num_beams
            kwargs["early_stopping"] = True
        elif method == "sample":
            kwargs["do_sample"] = True
            kwargs["top_k"] = 0
        elif method == "top_k":
            kwargs["do_sample"] = True
            kwargs["top_k"] = 50
        else:
            raise ValueError

        self.eval()
        with torch.no_grad():
            prediction_ids = self.generate(**kwargs)
        return prediction_ids

    @abstractmethod
    def get_hidden_states(self, encoder_outputs):
        pass

    @abstractmethod
    def get_total_loss(self, recon_loss: torch.Tensor, encoder_outputs):
        """
        Return (loss, reg_loss), with
            - loss: the total loss (objective to minimize)
            - reg_loss: the regularization loss (which can be None), e.g. KL div or MMD
        """
        pass


class GT8NonVAE(GT8Base):
    encoder_cls = T5Stack

    def get_hidden_states(
        self, encoder_outputs: BaseModelOutputWithPastAndCrossAttentions
    ):
        return encoder_outputs.last_hidden_state

    def get_total_loss(self, recon_loss: torch.Tensor, encoder_outputs):
        return recon_loss, None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        inputs = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
        return inputs


class GT8FullVAE(VAEBase, GT8Base):
    encoder_cls = VariationalT5Encoder

    def __init__(
        self,
        config,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_graph_token_id: int,
        reg_loss: Optional[str] = None,
    ):
        GT8Base.__init__(
            self,
            config,
            specify_target_with_prefix=specify_target_with_prefix,
            generate_text_token_id=generate_text_token_id,
            generate_graph_token_id=generate_graph_token_id,
        )
        self.reg_loss_type = reg_loss

    def get_hidden_states(self, encoder_outputs: VariationalT5EncoderOutput):
        return encoder_outputs.vae_latent

    def get_total_loss(self, recon_loss: torch.Tensor, encoder_outputs):
        reg_loss = self.compute_reg_loss(
            q_phi=encoder_outputs.q_phi, z=encoder_outputs.vae_latent
        )
        # loss = -L_elbo = -log p(x|z) + beta * reg_loss
        # where reg_loss can be
        #   - KL(q(z|x) || p(z)) (regular VAE)
        #   - MMD(q(z) || p(z)) (MMD VAE)
        return recon_loss + self.beta * reg_loss, reg_loss

    def compute_reg_loss(self, q_phi: Optional[Normal], z: torch.Tensor):
        if q_phi is None:
            # CycleVAE loss dual: we are computing the second reconstruction term only (e.g. log p(y_hat|z)
            # with previously computed z~q(z|y_hat)), we remove the regularisation term
            return 0.0

        # in all other cases, compute regularization loss as usual
        return super().compute_reg_loss(q_phi, z)

    @staticmethod
    def compute_kernel(x: torch.Tensor, y: torch.Tensor):
        """
        Compute the k(x,y) values with latent variable samples x,y. All combinations of x and y
        across batch dim are considered, but not across sequence dim, for memory reasons.

        Input: (N, T, dim)
        Output: (N, N, T)
        """
        N, T, dim = x.shape
        assert x.shape == y.shape

        # having (N*T)**2 samples with T=256 is impossible (50GB in memory)
        # so we consider latent samples across time dimension independently, and we use N**2 * T samples
        tiled_x = x.view(N, 1, T, dim).repeat(1, N, 1, 1)
        tiled_y = y.view(1, N, T, dim).repeat(N, 1, 1, 1)

        # compute RBF kernel k(x,y), shape: (N, N, T)
        sigma_sqr = dim ** 2 / 2
        squared_dist_xy = torch.sum((tiled_x - tiled_y) ** 2, dim=-1)
        return torch.exp(-0.5 * squared_dist_xy / sigma_sqr)


class GT8StyleVAE(VAEBase, GT8Base):
    encoder_cls = StyleVAET5Encoder

    def __init__(
        self,
        config,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_graph_token_id: int,
        reg_loss: Optional[str] = None,
    ):
        GT8Base.__init__(
            self,
            config,
            specify_target_with_prefix=specify_target_with_prefix,
            generate_text_token_id=generate_text_token_id,
            generate_graph_token_id=generate_graph_token_id,
        )
        self.reg_loss_type = reg_loss

    def get_hidden_states(self, encoder_outputs):
        hidden_states = encoder_outputs.last_hidden_state  # (N, T, dim)
        # replace the first token (initially the encoder representation of the
        # [STYLE] token) with the latent style variable s_x or s_y
        # Note: cloning is necessary to avoid inplace operations
        hidden_states[:, 0] = encoder_outputs.vae_latent.clone()
        return hidden_states

    def get_total_loss(self, recon_loss: torch.Tensor, encoder_outputs):
        reg_loss = self.compute_reg_loss(
            q_phi=encoder_outputs.q_phi, z=encoder_outputs.vae_latent
        )
        return recon_loss + self.beta * reg_loss, reg_loss

    @staticmethod
    def compute_kernel(x: torch.Tensor, y: torch.Tensor):
        """
        Compute the k(x,y) values with latent variable samples x,y. All combinations of x and y
        are considered here (since there is a single latent variable per sequence -> no sequence dimension).

        Input: (N, dim)
        Output: (N, N)
        """
        N, dim = x.shape
        assert x.shape == y.shape

        tiled_x = x.view(N, 1, dim).repeat(1, N, 1)
        tiled_y = y.view(1, N, dim).repeat(N, 1, 1)

        # compute RBF kernel k(x,y), shape: (N, N)
        sigma_sqr = dim ** 2 / 2
        squared_dist_xy = torch.sum((tiled_x - tiled_y) ** 2, dim=-1)
        return torch.exp(-0.5 * squared_dist_xy / sigma_sqr)

    def generate_with_target(
        self,
        *args,
        **kwargs,
    ):
        # add the 'source' kwarg to the generation (will be passed to encoder)
        if kwargs["target"] == "text":
            kwargs["source"] = "graph"
        elif kwargs["target"] == "graph":
            kwargs["source"] = "text"
        else:
            raise ValueError

        return super().generate_with_target(*args, **kwargs)


class GT8AddStyleVAE(VAEBase, GT8Base):
    encoder_cls = AddedStyleVAET5Encoder

    def __init__(
        self,
        config,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_graph_token_id: int,
        reg_loss: Optional[str] = None,
    ):
        super().__init__(
            config,
            specify_target_with_prefix=specify_target_with_prefix,
            generate_text_token_id=generate_text_token_id,
            generate_graph_token_id=generate_graph_token_id,
        )
        self.reg_loss_type = reg_loss

    def get_hidden_states(self, encoder_outputs: VariationalT5EncoderOutput):
        last_hidden_state = (
            encoder_outputs.last_hidden_state
            + encoder_outputs.vae_latent.unsqueeze(1).expand_as(
                encoder_outputs.last_hidden_state
            )
        )

        return last_hidden_state

    def get_total_loss(self, recon_loss: torch.Tensor, encoder_outputs):
        reg_loss = self.compute_reg_loss(
            q_phi=encoder_outputs.q_phi, z=encoder_outputs.vae_latent
        )
        # loss = -L_elbo = -log p(x|z) + beta * reg_loss
        # where reg_loss can be
        #   - KL(q(z|x) || p(z)) (regular VAE)
        #   - MMD(q(z) || p(z)) (MMD VAE)
        return recon_loss + self.beta * reg_loss, reg_loss

    # MMD-VAE specific methods
    @staticmethod
    def compute_kernel(x: torch.Tensor, y: torch.Tensor):
        """
        Compute the k(x,y) values with latent variable samples x,y. All combinations of x and y
        are considered here (since there is a single latent variable per sequence -> no sequence dimension).

        Input: (N, dim)
        Output: (N, N)
        """
        N, dim = x.shape
        assert x.shape == y.shape

        tiled_x = x.view(N, 1, dim).repeat(1, N, 1)
        tiled_y = y.view(1, N, dim).repeat(N, 1, 1)

        # compute RBF kernel k(x,y), shape: (N, N)
        sigma_sqr = dim ** 2 / 2
        squared_dist_xy = torch.sum((tiled_x - tiled_y) ** 2, dim=-1)
        return torch.exp(-0.5 * squared_dist_xy / sigma_sqr)

    def generate_with_target(
        self,
        *args,
        **kwargs,
    ):
        # add the 'source' kwarg to the generation (will be passed to encoder)
        if kwargs["target"] == "text":
            kwargs["source"] = "graph"
        elif kwargs["target"] == "graph":
            kwargs["source"] = "text"
        else:
            raise ValueError

        return super().generate_with_target(*args, **kwargs)
