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