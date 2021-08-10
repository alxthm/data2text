import copy
import warnings
from typing import Tuple, Optional

import torch
from dataclasses import dataclass
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, PreTrainedTokenizer, T5Config
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.utils import logging
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from src.data.formatting import add_prefix

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
        mu_z
        log_sigma_z
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # mean and log_std of the variational posterior (VAE encoder)
    mu_z: torch.FloatTensor = None
    log_sigma_z: torch.FloatTensor = None


@dataclass
class GT8ModelOutput:
    """
    Same attributes as the BaseModelOutputWithPastAndCrossAttentions class (output of the regular
    encoder, T5Stack)

    And some attributes from the VAE model:
        recon_loss: reconstruction loss (identical to loss, if we don't use VAE)
        kl_div
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

    recon_loss: Optional[torch.FloatTensor] = None
    kl_div: Optional[torch.FloatTensor] = None


class VariationalT5Encoder(T5PreTrainedModel):
    def __init__(self, t5_encoder: T5Stack):
        super().__init__(t5_encoder.config)
        self.t5_encoder = t5_encoder

        # additional layers for the variational posterior
        model_dim = t5_encoder.config.d_model
        self.mu_z = nn.Linear(model_dim, model_dim)
        self.log_sigma_z = nn.Linear(model_dim, model_dim)

    def forward(self, **kwargs):
        # make sure the output is always BaseModelOutputWithPastAndCrossAttentions
        assert kwargs["return_dict"]
        encoder_outputs = self.t5_encoder(**kwargs)
        mu_z = self.mu_z(encoder_outputs.last_hidden_state)
        log_sigma_z = self.log_sigma_z(encoder_outputs.last_hidden_state)

        return VariationalT5EncoderOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            mu_z=mu_z,
            log_sigma_z=log_sigma_z,
        )


class GT8(T5PreTrainedModel):
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

    def __init__(
        self,
        config,
        use_vae: bool,
        specify_target_with_prefix: bool,
        generate_text_token_id: int,
        generate_graph_token_id: int,
    ):
        super().__init__(config)
        self.model_dim = config.d_model
        self.use_vae = use_vae

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        if use_vae:
            # the same T5 encoder, but augmented with variational parameters
            self.vae_encoder = VariationalT5Encoder(self.encoder)

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

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

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
        if self.use_vae:
            return self.vae_encoder
        else:
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
        return_dict=True,
        target=None,
        vae_z=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.use_vae:
            # if we use the vae, vae_z should be either specified (during prediction),
            # either sampled (during training)
            assert (encoder_outputs is None and vae_z is None) or (
                encoder_outputs is not None and vae_z is not None
            )
        else:
            assert vae_z is None
        # make things easier to read by using ModelOutputs objects for encoder/decoder outputs
        # -> just make sure this is never overridden to False
        assert return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass(?))
        if encoder_outputs is None:
            encoder = self.vae_encoder if self.use_vae else self.encoder
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.use_vae:
                # during training, sample z from variational posterior
                q_phi = Normal(
                    loc=encoder_outputs.mu_z,
                    scale=torch.exp(encoder_outputs.log_sigma_z),
                )
                vae_z = q_phi.rsample()  # same dimension as mu_z (N, T, model_dim)

        # to be fed to the decoder
        if vae_z is None:
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = vae_z

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss, recon_loss, kl_div = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            recon_loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            loss = recon_loss

            if self.use_vae:
                # add the KL divergence term
                prior = Normal(
                    loc=torch.zeros_like(encoder_outputs.mu_z),
                    scale=torch.ones_like(encoder_outputs.mu_z),
                )
                kl_div = kl_divergence(q_phi, prior)  # (N, T, model_dim) as well
                # reduce to a scalar by:
                #   - summing over latent dim, to obtain the D_KL between multivariate Normal distributions
                #   - taking the mean over batch and sequence dim, to match the
                #       CrossEntropyLoss (which takes mean over N and T as well)
                kl_div = kl_div.sum(dim=2).mean()
                loss -= kl_div

        return GT8ModelOutput(
            loss=loss,
            kl_div=kl_div,
            recon_loss=recon_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

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

        if self.use_vae:
            if "vae_z" in kwargs:
                # use vae_z specified as an kwarg to the generate method
                vae_z = kwargs["vae_z"]
            else:
                # take mu_z (but we could also sample)
                vae_z = encoder_outputs.mu_z
            inputs["vae_z"] = vae_z

        return inputs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor, target: str):
        # in principle, this method should not be called
        return self._shift_right(labels, target)

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
    ):
        """

        Args:
            input_ids:
            target: 'text' or 'graph'
            tokenizer:
            max_seq_length:
            method: 'greedy', 'beam_search', 'sample' or 'top_k'
            num_beams: Used only when method='beam_search'

        Returns:

        """
        # specify the target format to the model
        if self.specify_target_with_prefix:
            input_ids = add_prefix(
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

        self.eval()
        with torch.no_grad():
            # generate text according to the specified decoding method
            if method == "greedy":
                prediction_ids = self.generate(
                    input_ids,
                    max_length=max_seq_length,
                    decoder_start_token_id=decoder_start_token_id,
                )
            elif method == "beam_search":
                assert num_beams > 1
                prediction_ids = self.generate(
                    input_ids,
                    max_length=max_seq_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    decoder_start_token_id=decoder_start_token_id,
                )
            elif method == "sample":
                prediction_ids = self.generate(
                    input_ids,
                    max_length=max_seq_length,
                    do_sample=True,
                    top_k=0,
                    decoder_start_token_id=decoder_start_token_id,
                )
            elif method == "top_k":
                prediction_ids = self.generate(
                    input_ids,
                    max_length=max_seq_length,
                    do_sample=True,
                    top_k=50,
                    decoder_start_token_id=decoder_start_token_id,
                )
            else:
                raise ValueError

        return prediction_ids
