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
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size

        # embeddings are being extracted patch by patch with nok overlapping
        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,  # hidden size  # mean series of embedding
            kernel_size=self.patch_size,
            stride=self.patch_size,  # all patches are non-overlapping
            padding="valid",  # no padding added
        )  # [ batch,size, no. of f patches, embed_dim]
        # image_size = 224, for baese version of poligemma
        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2  # **2 because of 2D image i.e. 16 * 16
        self.num_positions = (
            self.num_patches
        )  # positional encidings are equal to number of patches becasue we need the inforrmation about where each patch is in the image.

        # [1, self.num_positions] -> [1, self.num_positions, self.embed_dim]
        self.position_embedding = nn.Embedding(
            self.num_positions, self.embed_dim
        )  # this vector is same size of partch embedding vector  # each of this will be added to patvh_embedding vector

        # [1, self.num_positions]
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = (
            pixel_values.shape
        )  # [batch_size, num_channels, height, width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipMLP(nn.Module):
    """
    to add non linearity and trainable parameters
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(
            self.config.hidden_size, config.intermediate_size
        )  # intermediate size is 3072.. 4 times of hidden size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, hidden_size] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, hidden_size]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    """Multi-head attention from 'Attention is All You Need' paper"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # equivalent to 1/sqrt(head_dim)
        self.dropout = config.attention_dropout

        # parameter matrices for key, query, value
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Wk
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Wv
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Wq
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Wo

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # k,q,v are just transformations of input sequence

        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, embed_dim = hidden_states.size()

        # query_states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)

        # key_states: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)

        # value_states: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)

        # we do this because the self-attention mehanism needs to see same sequence in three different ways as k,q and v

        # split each token into smaller tkoens based on number of heads
        # each head focus on different part of the sequence and each head is independent too

        # query_states: [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, num_heads=8, head_dim] -> [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # calculate the attention scores using the scaled dot-product method formula : Q.K^T / sqrt(d_k)
        # [batch_size, num_heads, num_patches, head_dim] * [batch_size, num_heads, head_dim, num_patches] -> [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual : [batch_size, num_patches, embed_dim/hidden_size]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim/hidden_size]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim/hidden_size]-> [batch_size, num_patches, embed_dim/hidden_size]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embed_dim/hidden_size]
        hidden_states = hidden_states + residual
        # residual : [batch_size, num_patches, embed_dim/hidden_size]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim/hidden_size] -> [batch_size, num_patches, embed_dim/hidden_size]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim/hidden_size] -> [batch_size, num_patches, embed_dim/hidden_size]
        hidden_states = self.mlp(hidden_states)  # mlp adds non-linearity and parameters
        # [batch_size, num_patches, embed_dim/hidden_size]
        hidden_states = hidden_states + residual
        return hidden_states


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
