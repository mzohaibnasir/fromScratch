from typing import Optional, Tuple
import torch
import torch.nn as nn

# poligemma


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,  # embedding size
        intermediate_size=3072,  # size of linear layer
        num_hidden_layers=12,  # number of layers of vision transformer
        num_attention_heads=12,  # number of heads in multihead attention
        num_channels=3,  # RGB
        image_size=224,
        patch_size=14,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,  # how many output embedding we will have for each image; each of these contextualized embedding will be considered as a tokens of image.It wont ba a one single embrding that represents whole imagebut list of embeddings that represesnt a patch of each image and als info about other patches throigh the attention mechanismo
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = (intermediate_size,)
        self.num_hidden_layers = (num_hidden_layers,)
        self.num_attention_heads = (num_attention_heads,)
        self.num_channels = (num_channels,)
        self.image_size = (image_size,)
        self.patch_size = (patch_size,)
        self.layer_norm_eps = (layer_norm_eps,)
        self.attention_dropout = (attention_dropout,)
        self.num_image_tokens = num_image_tokens


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(
            config
        )  # pacthes will be converted to embeddings
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel values: [batch_size, num_channels, height, width] => [batch_size, num_image_tokens/num_patches, hidden_size/embedding_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """
        [batch_size, num_channels, height, width] => [batch_size, num_image_tokens/num_patches, hidden_size/embedding_dim]
        .
        .

        takes in batch of images and returns list of embeddigs for each image in that batch
        """

        return self.vision_model(pixel_values=pixel_values)
