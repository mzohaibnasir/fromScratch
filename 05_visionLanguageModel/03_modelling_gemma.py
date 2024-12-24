# language model
"""
Gemma in Paligemma, the model that will generate textual outputs to input prompt conditioed on inoput image
This class connects are components including vision encoder, language model and multi modal projector, transformer decoder and input text encoder
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from torch.nn import CrossEntropyLoss
import math
from 01_modelling_siglip import SiglipVisionConfig, SiglipVisionModel



class PaliGemmaConfig():
    def __init__(
            self,
            vision_config = None,# vision encoder config
            text_config = None, # gemma config..lang_model
            ignore_index = -100,
            image_token_index = 256000, #<image> id
            vocab_size = 257152,
            projection_dim = 2048, # dim visoin encoder output should be resized to (linear Projection layer output size)
            hidden_size = 2048, # language model embedding size
            pad_token_id = None,
            **kwargs                 
    ):
        super().__init__()
        self.ignore_index = ignore_index,
        self.image_token_index = image_token_index,
        self.vocab_size = vocab_size,
        self.projection_dim = projection_dim,
        self.hidden_size = hidden_size,
        self.vision_config = vision_config,
        self.is_encoder_decoder = False,
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config


        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)**2
        self.vision_config.projection_dim = projection_dim


        

# first create structure if model than compoentns



class PaliGemmaForConditionalGeneration(nn.Module):
    """
    conditional generation model that generates text based on input image. howerver it is slso because of attention mask that we create while generation
    """
    def __ini__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config) # contrastive vision encoder
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config) # linear projection  after vision encoder
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config) # transformer decoder
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1



    def tie_weights(self):
        """
        means that certain layers or components of the model share the same set of parameters (weights). 
        This technique is often used to reduce the number of parameters in the model and ensure consistency between related layers.
        for example; the input embeddings(output embedding that converts user input to embedding and Linear layer that converts contextualized embedding to vocab size) and output embeddings are doing opposite tasks.. input of the model share the same weights
        Weight tying shares the weights between the input embedding layer and output projection layer in neural language models.
        Instead of learning separate matrices for converting tokens to embeddings and projecting hidden states to logits, the same matrix is used for both. 
        This reduces model size by ~20-25% and often improves performance through better regularization and more efficient learning."""
        return self.language_model.tie_weights()


    def forward(self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.tensor] = None,
                kv_cache: Optional[KVCache] = None,
                )-> Tuple:
        assert torch.all(attention_mask ==1 ), "the input cannot be padded"


        # 1. extract the input embeddings
        # embeddings : <image><bos><prompt><\n>
        # shape:(batch, seq_len, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids) # as we are using placeholder <image> for image tokens. so toeknizer will return token id for <image> token but as we know its just a placeholder so we will replace it with image embedding later.


        # 2. Merge text and image embeddings
        # images: [batch, channel, height, width] -> [batch, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype)) 


        # resize image embeddings inot same size of language model embeddings.

        # [batch_Size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size)]
        image_features = self.multi_modal_projector(selected_image_feature)


        # now we need to merge the tokens extracted form vision model and text tokens which already contains placeholders where we should put our image embeddings
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, # from vision model
                                                                                               input_embeds, # tet embeddings containing place holders
                                                                                               input_ids, # 
                                                                                               attention_mask,
                                                                                               kv_cache)


        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache
        )

        return outputs
                                                                                              