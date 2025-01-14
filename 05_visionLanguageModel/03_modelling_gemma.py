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






class KVCache():
    def __init__(self) -> None:
        self.key_cache=[]
        self.value_cache=[]
        
    def num_items(self)-> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # rember we are adding key_states and value_states to key_cache and value_cache
            # which are of the shape [batch_size, num_heads_kv seq_len, head_dim]
            # so we would be returing seq_len currently stored in kv_cache

            # For each token: 
            # Keys and values are appended to the KV cache along the sequence length dimension.
            # At each decoding step, the KV cache grows by one entry along the sequence length dimension.
            return self.key_cache[0].shape[-2]


    def update(self, 
               key_states : torch.Tensor,
               values_states : torch.Tensor, 
               layer_idx: int,
               )-> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # if we never added anything to the KV-Cache of this layer, let's craetea it
            self.key_cache.append(key_states)
            self.value_cache.append(values_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [batch_size, num_heads_kv, seq_len, head_dim]

            self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim =-2
                                                            )
            self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], values_states], dim =-2
                                                        )
        # ... and then we return all the existing keys  + new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
            





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


        
class PaliGemmaMultiModalProjector(nn.Module):
    """converts hidden size of vision model to into projection dimension which is embedding size of text model"""
    def __init__(self, config: PaliGemmaConfig):
         super().__init__()
         self.linear = nn.Linear(
              config.vision_config.hidden_size,
              config.vision_config.projection_dim,
              bias=True
         )
    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] => [batch_size, num_patches, projection_dim]
        hidden_state = self.linear(image_features)
        return hidden_state



class GemmaRotaryEmbedding(nn.Module):
    def __init__(self,
                 dim, 
                 max_position_embeddings=2048, # max number of positions we can encode
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim # it is set to head_dim # as rope mpodify attention mechansim and attention mechansim is performed independently for each heaad so each head will hav eits own postional encoding applied to.
        self.max_position_embeddings =  max_position_embeddings
        self.base = base


        # calculate  the theta according to  the formula theta_i =base^(2i/dim) where i =0,1,....., dim//2
        inv_freq = 1.0/(self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float()/ self.dim))
        self.register_buffer(
             "inv_freq",
             tensor= = inv_freq,
             persistent=False
        )




class GemmaRMSNorm(nn.Module):
    def __init__(
            self,
            dim:int, 
            eps:float= 1e-6
    ):
        super().__init__()
        self.eps = eps  # as we are multiplying by 1/sqrt(..), if ,so if (sqrt(...)) is too small, it would give really big number.. we avoid this  using eps
        self.weight == nn.Parameter(torch.zeros(dim)) # this is that gamma(learnable paramter) # number of parameters, one for each fature
    
    def _norm(self, x):
        return x* torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps # 1/sqrt(...)
        )

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    """
    gated projections, adds more leaenable parameters........ also used in llama
    to add non linearity and trainable parameters
    first layer expands and then reduces dims gives oppotunity to learn complex features
    """


    def __init__(self, config):
        super().__init__())

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # this is  used foe activation function that gemma is using
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


    def forward(self, x):
        # # equivalen to 
        # #[batch_size, seq_len, hidden_size] -> [ batch_size, seq_len. intermediate_size]
        # y = self.gate_proj(x)
        # # [batch_size. seq_len, intermediate_size]
        # y = torch.gelu(y, approximate='tanh')


        # # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        # j = self.up_proj(x)
        
        # # [batch_size, seq_len, intermediate_size]
        # z= y* j 

        # # [batch_size, seq_len. intermediate_size]-> [batch_size, seq_len, hidden_size]
        # z= self.down_proj(z)


        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate='tanh')
            *
            self.up_proj(x)
        )
    

def repeat_kv(
          hidden_states: torch.Tensor,
          n_rep:int
    )-> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep ==1:
         return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch,num_key_value_heads,n_rep, slen,head_dim)
    return hidden_states.reshape(batch, num_key_value_heads*n_rep, slen, head_dim)




class GemmaAttention(nn.Module):

    def __init__(self,
                 config: GemmaConfig,
                 layer_idx: Optional[int] =  None
                 ):

                 super().__init__()
                 self.config = config
                 self.layer_idx = layer_idx

                 self.attention_dropout = config.attention_dropout
                 self.hidden_size = config.hidden_size
                 self.num_heads = config.num_attention_heads
                 self.head_dim = config.head_dim # num of attention heads for query
                 self.num_key_value_heads = config.num_attention_heads
                 self.num_key_value_groups = self.num_heads // self.num_key_value_heads
                 self.max_position_embeddings = config.max_position_embeddings
                 self.rope_theta = config.rope_theta
                 self.is_causal = True

                 assert self.hidden_size % self.num_heads ==0 
                 

                 # In group query attention: numbers of heads in query are more than k and v heads
                # num_heads = 8
                 # hidden_size = 1024
                 # hed_dim = hidden_size//num_heads = 128
                 # Wq: [1024, 8*128]  = [1024, 1024]
                 # Wk: [1024, 1*128] = [1024, 128]
                 # Wv: [1024, 1*128] = [1024, 128]
                 # Wo:[1024, 1024]
                 self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)
                 self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
                 self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
                 self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.num_key_value_heads * self.hidden_size, bias=config.attention_bias)

                 # rotary position encoding
                 self.rotary_embed = GemmaRotaryEmbedding(
                      self.head_dim,
                      max_position_embeddings = self.max_position_embeddings,
                      base = self.rope_theta
                 )


    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                **kwargs,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                
                # [batch_size, seq_len, hidden_size]
                bsz, q_len, _ = hidden_states.size()

                #[batch_size, seq_len, num_heads_q * head_dim]
                query_states = self.q_proj(hidden_states)

                
                #[batch_size, seq_len, num_heads_kv * head_dim]

                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                # [batch_size, num_heads_Q, seq_len, head_dim]
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)

                # [batch_size, num_heads_KV, seq_len, head_dim]
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)


                # apply roatary position encoding
                # we are just modifying these embeddings by adding some positional information
                #[batch_size, seq_len, heqd_dim] -> [batch_size, seq_len, head_dim]
                cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
                
                #[batch_size, num_heads_q, seq_len, head_dim] -> [batch_size, num_head_kv, seq_len, head_dim]
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)


                # when woring with kv_cache , we only pass one token as input to the transormer layers and then this single otken is added to kv_cache of this particular laye 
                # and then we retrieve the content of this kv_cache whoch also includesthe current token an dthen we use this kv_cahe to compute attention

                if kv_cache is not None:
                    # for prefilling, all keys and value will go in at once but for inference one at a time
                    key_states, value_states = kv_cache.update(
                         key_states, # token we are adding to key_cache
                         value_states, # token we are adding to value_kache
                         self.layer_idx
                    )

                # repeat the key and values to match  the number of heads and query
                # not actually using GQA i.e. head_q = head_kv
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                # head_q ==head_v ==head_k


                # perform the calculation as usual, Q* K^T /sqer(head_dim). shape :[batch_size, num_heads_q, seq_len_q, seq_len_kv]
                attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(self.head_dim)

                assert attention_mask is not None, "Attention mask is required"

                attn_weights = attn_weights + attention_mask  # here attn amasks will be all 0s as we dont have paddings and we are not masking anything in prfeilling  as input prompt also attends future tokens


                # apply the softmax
                # [batch_size, num_heads_q, seq_len_q, seq_len_kv] # row by row
                attn_weights = nn.functional.softmax(attn_weights,dim=-1, dtype=torch.float32).to(query_states.dtype)

                # add dropout
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

                # multiply by V. [batch_size, num_heads_q, seq_len_q, seq_len_kv] *[batch_size, num_heads_kv, seq_len_kv, head_dim] -> [batch_size, num_heads_q, seq_len_q, head_dim]
                attn_output = torch.matmul(attn_weights, value_states) # weighted sum: each token is aggregation of all the previous tokens based on the score defined by attention matrix
                 
                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"Attention output size {attn_output.size()} does not match {(bsz, self.num_heads, q_len, self.head_dim)}"
                    )

                # [batch_size, num_head_q, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads_q, head_dim]
                attn_output = attn_output.transpose(1,2).contiguous()

                # [batch_size, num_head_q, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads_q * head_dim]
                attn_output = attn_output.view(bsz, q_len, -1 )

                # Multiply by Wo, [batch_size, seq_len_q, hidden_size   ]
                attn_output =self.o_proj(attn_output)


                return attn_output, attn_weights






                



                


# same as silip
class GemmaDecoderLayer(nn.Module):
    def __init__(self, 
                 config: GemmaConfig,
                 layer_idx: int
                 ):
    
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(
            config=config,
            layer_idx=layer_idx
        )
        self.mlp = GemmaMLP(config) # FFN
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps = config.rms_norm_eps   
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps = config.rms_norm_eps   
        )

    def forward(
            self,
             hidden_states: torch.Tensor, 
             attention_mask: Optional[torch.Tensor]= None,
             position_ids: Optional[torch.LongTensor]= None,
             kv_cache : Optional[KVCache] = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        #[batch_size, sql_len,  hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        #[batch_size, sql_len,  hidden_Size]
        hidden_states, _ , = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache
        )

        #[batch_size, sql_len,  hidden_Size]
        hidden_states = hidden_states + residual

        
        #[batch_size, sql_len,  hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)


        #[batch_size, sql_len,  hidden_Size]
        hidden_states = self.mlp(hidden_states)


        #[batch_size, sql_len,  hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states





















class GemmaModel(nn.Module):

    """
        Language model is an embeddings layer, series of transfomer layers and then the lamguage  modelling head.
        #LM_head is already implemented in GemmaForCausalLM
    """
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size


        self.embed_tokens = nn.Embedding(
             config.vocab_size,
             config.hidden_size,
             self.padding_idx
        )

        self.layers = nn.ModuleList(
             [GemmaDecoderLayer( config, layer_idx ) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def get_input_embeddings(self):
        return self.embed_tokens # to extract initial embeddings


    def forward(
              self,
              attention_masks: Optional[torch.Tensor] = None,
              position_ids: Optional[torch.LongTensor] = None, #  used to apply rotary positional embeddings.. here positional embeddings are applied just before applying attention .. unlike vanilla transformer
              input_embeds: Optional[torch.FloatTensor] = None,
              kv_cache: Optional[KVCache] = None
              )-> torch.FloatTensor:
        # [ batch_size, seq_len, hidden_size]
        hidden_states =input_embeds

        # [ batch_size, seq_len, hidden_size]
        normalizer = torch.tensor(
        self.config_hidden_size **0.5, dtype=hidden_states.dtype
        )
        
        hidden_states = hidden_states* normalizer


        for decoder_layer in self.layers:
            #[batch_size, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_masks=attention_masks,
                position_ids=position_ids,  
                kv_cache=kv_cache
            )

        # [batch_size, seq_len, hidden_size]            
        hidden_states  = self.norm(hidden_states)

        # [batch_size, seq_len, hidden_size]
        return hidden_states











# first create structure if model than compoentns
class GemmaForCausalLM(nn.Module):
    """xxCausalLM always means it is transformer model with language modeling head i.e. self.lm_head"""
    def __init__(self, config):
          super().__init__()
          self.config = config
          self.model = GemmaModel(config)
          self.vocab_size = config.vocab_size
          self.lm_head = nn.Linear(
               config.hidden_size,
               config.vocab_size,
               bias = False
          )

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        """
        sharing weights aming embedding layer and logits layer"""
        self.lm_head.weight = self.model.embed_tokens.weight
    
    
    def forward(
            self,
            attention_mask: Optional[torch.Tensor]= None,
            position_ids : Optional[torch.LongTensor] = None,
            inputs_embeds : Optional[torch.FloatTensor] =  None,
            kv_cache : Optional[KVCache] = None,
    ):
        # input_embds: [batch_size, seq_len, hidden_size]
        # outputs: [batch_size, seq_len, hidden_size]

    
        # send to model and outputs will be embeddings
        outputs = self.model(
              attention_mask = attention_mask,
              position_ids = position_ids,
              inputs_embeds = inputs_embeds,
              kv_cache = kv_cache   
         )


        # but we do not want embeddings, we want LOGITS
        hidden_state = outputs
        logits = self.lm_head(hidden_state)
        logits = logits.float()

        return_data = {
             "logits":logits
        }

        # if user specified kv_cahe, we retunrthat too

        if kv_cache is not None:
            # return the updated cache
            return_data['kv_cache'] = kv_cache
        return return_data

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
            text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim) # -1 mean dont keep this dim same
            pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim) # -1 mean dont keep this dim same
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
            # BLOCK attention throughout image and prefix  and AUTOREGRESSIVE attention on the suffix

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
            # we need the positions of tokens that would be used by rotary position encodings


            if kv_cache is not None and kv_cache.num_items() >0: 
                 #prefilling
                 # the position of qurey is just the last position
                 # this will be used to assess which rotary positional encdogin we are going to apply to each token
                 position_ids = attention_mask.cumsum(-1)[:,-1]# it should be equal to number of tokens in prompt.. as there are only 1s in attention_mask and no padding tokens so we can directly use them
                 if position_ids.dim() == 1:
                      position_ids = position_ids.unsqueeze(0)
            else:
                 # token generation: now wehave one single query to apply positional encoding and for that we only take one token
                 # create a position_ids baed on current  size of attention_mask
                 # for masked tokens, use number 1 as position.


                 # when we generate tokens, basically we have some tokens akready in kv_cache and then we have one new token which  is last predict token
                 # which we use as a query. To  understand what is position of this token, we also provide attention mask. Attention mask indicates 
                 # that it's all made up of 1s. how many 1s? tokens in kv_caache :n+ new token :1...noew token that we need to add to kvacache before we calculate attention
                 #  ..so here attention_mask.cumsum(-1) we are counting tokens in kva_cache


                 position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask==0), 1).to(device=device) # 

            return final_embedding, causal_mask, position_ids


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
                                                                                              