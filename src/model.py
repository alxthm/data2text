import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5PreTrainedModel,
    T5Config,
)
from src.data.formatting import add_prefix

class T5Custom(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)

        self.t5_for_conditional_generation=T5ForConditionalGeneration(config)
        
       
       # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.t5_for_conditional_generation.init_weights()


    def deparallelize(self):
        self.t5_for_conditional_generation.deparallelize()
        self.t5_for_conditional_generation = self.t5_for_conditional_generation.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.t5_for_conditional_generation.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        
        self.t5_for_conditional_generation.parallelize(device_map)
        self.model_parallel = True
    
    def get_input_embeddings(self):
        return self.t5_for_conditional_generation.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.t5_for_conditional_generation.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.t5_for_conditional_generation.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.t5_for_conditional_generation.get_output_embeddings()

    def get_encoder(self):
        return self.t5_for_conditional_generation.get_encoder()

    def get_decoder(self):
        return self.t5_for_conditional_generation.get_decoder()


    def forward(self, *args , **kwargs ):
        return self.t5_for_conditional_generation.forward(*args, **kwargs)

    def predict_(self, input_ids: torch.Tensor, target: str, tokenizer:PreTrainedTokenizer,max_seq_length: int): 
        input_ids = add_prefix(
            input_ids=input_ids,
            target=target,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        )
        
        self.eval()
        with torch.no_grad():
            prediction_ids = self.generate(
                input_ids,
                max_length=max_seq_length,
                num_beams=1,
            )
        
        return prediction_ids


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
        **kwargs
        ):
        return self.t5_for_conditional_generation.prepare_inputs_for_generation(
            input_ids, 
            past=past,
            attention_mask=attention_mask,
            head_mask= head_mask,
            decoder_head_mask =decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            **kwargs)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.t5_for_conditional_generation.prepare_decoder_input_ids_from_labels(labels)


    def _reorder_cache(self, past, beam_idx):
        return self.t5_for_conditional_generation._reorder_cache(past, bean_idx)
            
