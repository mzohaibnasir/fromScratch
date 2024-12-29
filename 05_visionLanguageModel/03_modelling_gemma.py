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

class GemmaConfig():
    def __init__(
            self,
            vocab_size,
            hidden_size, # embedding vector size
            intermediate_size, # intermediate size of ff layer
            num_hidden_layers, # number of layers transformer has
            # group query attention : different number of head for query and different number of heads for key and value
            num_attention_heads, # number of query heads
            num_key_value_heads, # number of key value heads
            head_dim=256, # d_head
            max_position_embeddings = 8192, # max number of postions  our model has been trained upon
            rms_norm_eps = 1e-6, # rms normalizaion
            rope_theta = 10000.0, # rotatory position encoding: base frequency
            attention_bias = False, # as we know wk, wq, wv are linear layers so we can have bias terms
            attention_dropout = 0.0,
            pad_token_id = None,
            **kwargs,
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.head_dim = head_dim
            self.num_key_value_heads = num_key_value_heads
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.attention_bias = attention_bias
            self.attention_dropout = attention_dropout
            self.pad_token_id = pad_token_id



class PaliGemmaConfig():
    def __init__(
            self,
            vision_config = None,# vision encoder config
            text_config = None, # gemma config..lang_model
            ignore_index = -100, # for labels while training
            image_token_index = 256000, #<image> id # imag placeholder token_id 
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

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)**2 # meaning how many pacthes we will get
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



    def _merge_input_ids_with_image_features(
              self, 
              image_features: torch.Tensor,
              input_embeds: torch.Tensor,
              input_ids: torch.Tensor,
              attention_mask: torch.Tensor,
              kv_cache: Optional[KVCache] =None

        ):
            # extract info from input
            _, _, embed_dim =image_features.shape
            batch_size, sequence_length = input_ids.shape  #[0] : I love you => 12 15 19 (tokenized)(batch:1, seq_len: 3)
            dtype, device = input_embeds.dtype, input_embeds.device

            #shape:[batch_size, seq_len, hidden_size]
            scaled_image_features = image_features / (self.config.hidden_size**0.5) # equivalent to 1/sqrt(head_dim) # same kind of scaling that we usein attention mechanism

            # combine the embeddings of the image tokens, the next tokens, and mask out all the padding tokens.
            final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device)

            #creating masks
            #shape: [batch, seq_len]
            
            # true foe text tokens
            text_mask = (input_ids != self.config.image_token_index) & (input_ids !=self.pad_token_id)
            #true for image tokens  # imahe placeholder tokens
            image_mask = input_ids == self.config.image_token_index
            # true for padding tokens
            pad_mask = input_ids == self.pad_token_id
            
            # so we will place image token, text token and padding token in this final_embedding based on these masks


            # we need to expand the masks to the embedding dimension otherwise we can't  use them in torch.where
            #[batch, seq_len] -> [batch, seq_len, embed_dim]
            text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim) # -1 mean dont leep this dim same
            pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim) # -1 mean dont leep this dim same
            image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim) # -1 mean dont leep this dim same


            # conirming no mask overlap
            assert not (text_mask & image_mask).any(), "Overlap detected between text and image masks."
            assert not (text_mask & pad_mask).any(), "Overlap detected between text and pad masks."
            assert not (image_mask & pad_mask).any(), "Overlap detected between image and pad masks."


            # [batch, seq_len, embed_dim]
            # first add text_embeddings
            final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)

            # insert image embeddings
            # we can't use torch.where because the sequence length of scaled_image_features is not equal to equal to the dequnce length  of final embedding.
            final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)  # place where we have placeholder token if for <image>

            # insert padding
            final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)


            # so noew we have [256 image tokenn embefdings +<text embeddng>]


            ## CREATING ATTENTION MASK

            # attention mask will be created based on how we are wokring with KV-Cache
            
            dtype, device = input_embeds.dtype, input_embeds.device
            min_dtype = torch.finfo(dtype=dtype)
            q_len = input_embeds.shape[1]


            if kv_cache is None or kv_cache.num_items() == 0:
                # 1. PREFILLING
                # Do not mask any token, becasue we are in prefill phase
                # this only works when you haveno padding
                causal_mask = torch.full(
                     (batch_size, q_len, q_len),
                     fill_value=0, # mask is made up of -inf for all the positions for whoch we dont want interactions..but here we are not using -inf
                     dtype=dtype,
                     device=device
                ) 

            else:   
                # 2. TOKEN GENERATION
                # since we are genreating tokens, the query must be one single token so q_len must be 1
                assert q_len == 1# generating one token at a time

                kv_len = kv_cache.num_items() + q_len # adding current query token
                # also in this case we dont need to mask anything, since each query should be able to attend all previous
                # this only works when we have no padding
                causal_mask = torch.full(
                     (batch_size, q_len, kv_len),
                     fill_value= 0,
                     dtype=dtype,
                     device=device
                )

            # add the head dimebsion
            # [batch_size, q_len, kv_len] -> [batch_size, num_head_Q, Q_len, kv_len]
            causal_mask = causal_mask.unsqueeze(1) # causal  mask is made up of -inf for all the positions for whoch we dont want interactions




            # kvcache coded



            # rotary position encodings
            # positions of tokens that would be used by rotary position encodings


            if kv_cache is not None and kv_cache.num_items() >0:
                 # the position of qurey is just the last position
                 position_ids = attention_mask.cumsum(-1)[:,-1]
                 if position_ids.dim() == 1:
                      position_ids = position_ids.unsqueeze(0)
            else:
                 # create a position_ids baed on current  size of attention_mask
                 # for masked tokens, use number 1 as position.


                 position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask==0), 1).to(device=device)


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
                                                                                              