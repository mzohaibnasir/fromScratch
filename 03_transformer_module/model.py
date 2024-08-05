import torch
import torch.nn as nn
import math
import numpy as np

# 1 input embeddings


class InputEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        print("\tsampleInputShape: (batch, seq_len) : ", x.shape)
        x = self.embedding(x)
        print("\tinputEmbeddingsReturnShape: (batch, seq_len, d_model) : ", x.shape)
        return x


# Example usage
# input_embeddings = InputEmbeddings(d_model=512, vocab_size=1000)
# Create an example input tensor (batch size 1, sequence length 5, embedding dimension 20)
# batch_of_sentences = torch.tensor(
# [[5, 6, 7, 0, 0]])  # Shape: (batch_size, max_sentence_length)
# print(batch_of_sentences.shape)

# Pass through the embedding layer
# The forward method is called automatically when you use the instance like a function.
# embedded_sentences = input_embeddings(batch_of_sentences)
# print(
#     "embedded_sentences.shape: ",
#     embedded_sentences.shape,
#     " embedded_sentences: ",
#     embedded_sentences,
# )  # (batch, seq_len, embedding dim)

######################################################################################################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # build matrix of (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #  create a vector of shape (seq_len,1) to represent position of word in sequence
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len,1)  # pos in formula

        # create denominator
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sin to even positions
        pe[:, 0::2] = torch.sin(pos * div_term)

        # apply cos to odd positions
        pe[:, 1::2] = torch.cos(pos * div_term)

        # print("pe.shape", pe.shape)
        pe = pe.unsqueeze(0)  # (batch, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        we need to add positional encoding to every token/word inside sequence/sentence
        """
        x = x + self.pe[:, : x.shape[1], :]  #
        # x:token and pe is positional encoding  # because we dont want to learn pe because these are fixed
        print("\tpositionalEncodingReturnShape: (batch, seq_len, d_model) : ", x.shape)

        #  :x.shape[1]:selecting just # of tokens because of input sequence length.
        return self.dropout(x)  # (batch, seq_len, d_model)


# Example usage
# positional = PositionalEncoding(d_model=512, seq_len=35, dropout=0.5)
# print("positional:", positional)

# Create an example input tensor (batch size , sequence length , embedding dimension )

# print("\n\n\n input to PE", embedded_sentences.shape)

# positional_encoded = positional(embedded_sentences)
# print("input ", embedded_sentences)

# print("input shape", embedded_sentences.shape)
# print("positional_encoded shape", positional_encoded.shape)

# print(positional_encoded)  # (1, seq_len,d_model)

######################################################################################################################


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # gamma  # mulltiplied
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        # print(x.shape)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # print("mean shape", mean.shape, mean)
        normalized = self.alpha * (x - mean) / (std + self.eps) + self.bias

        print("\tnormalizedReturnShape: (batch, seq_len, d_model) : ", normalized.shape)

        return normalized


# ln = LayerNormalization()

# print("Before normalization:")
# print(positional_encoded)

# normalized = ln(positional_encoded)
# print("After normalization:")
# print(normalized.shape)
# normalized  # (1, seq_len,d_model)

##########################################################################################

# feed forward network


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.5):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        # print("\tbeforeFeedForwardReturnShape: (batch, seq_len, d_model) : ",
        #       x.shape)

        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        print("\tfeedForwardReturnShape: (batch, seq_len, d_model) : ", x.shape)

        return x


# feedforwardblock = FeedForwardBlock()

# feedforwarded = feedforwardblock(normalized)

#############################################################################################

# Multi


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        attention_scores = None

        # weight matrices
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # output matrix :Wo (h*dv, d_model)
        self.wo = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # print(
        #     "\n\n\n\t\t\t\t\t\t MHA().attention called:\n\t\t\t\t\t\t\tattentionReturnShape: (batch,heads, seq_len, d_k) : ",
        #     attention_scoresV.shape,
        # )

        print(
            "\n\n\n\t\t\t\t\t\tAttention called:\n\t\t\t\t\t\t\tquery: (batch, heads, seq_len, d_model) : ",
            query.shape,
        )
        print(
            "\t\t\t\t\t\t\tkey: (batch, heads, seq_len, d_model) : ",
            key.shape,
        )
        print(
            "\t\t\t\t\t\t\tvalue: (batch, heads, seq_len, d_model) : ",
            value.shape,
        )

        # print("\t\t\t\t\t\tkey: (batch, seq_len, d_model) : ", key.shape)
        # print("\t\t\t\t\t\tvalue: (batch, seq_len, d_model) : ", value.shape)

        d_k = query.shape[-1]  # extract embedding length

        # (b,h,seq_len,dk)->b,h,seq_len,seq_len)
        attention_scores = torch.einsum("bhij,bhkj->bhik", query, key) / math.sqrt(d_k)

        if mask is not None:
            print(
                "\t\t\t\t\t\t\tmask: (batch, heads, seq_len, d_model) : ",
                mask.shape,
            )
            # replace all values with very small number so softwax will assign them 0 in output.
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (b,h,seq_len, seq_len) -> (b,h, seq_len, d_k)
        attention_scoresV = torch.einsum("bhsj, bhsk->bhsk", attention_scores, value)

        print(
            "\t\t\t\t\t\tAttentionScoreShape: (batch, heads, seq_len, seq_len) : ",
            attention_scores.shape,
        )
        print(
            "\t\t\t\t\t\tAttentionReturnShape: (batch, heads, seq_len, d_k) : ",
            attention_scoresV.shape,
        )

        print("\n")
        return (attention_scoresV, attention_scores)

    def forward(self, q, k, v, mask):
        query = self.wq(q)
        value = self.wv(v)
        key = self.wk(k)
        print(
            "\n\n\n\t\t\t\t\t MHA() called:\n\t\t\t\t\t\tquery: (batch, seq_len, d_model) : ",
            query.shape,
        )

        print("\t\t\t\t\t\tkey: (batch, seq_len, d_model) : ", key.shape)
        print("\t\t\t\t\t\tvalue: (batch, seq_len, d_model) : ", value.shape)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)

        print(
            "\n\n\n\t\t\t\t\t MHA() called: After splitting\n\t\t\t\t\t\tquery: (batch, seq_len, head, d_model) : ",
            query.shape,
        )

        print("\t\t\t\t\t\tkey: (batch, seq_len, head, d_model) : ", key.shape)
        print("\t\t\t\t\t\tvalue: (batch, seq_len, head, d_model) : ", value.shape)

        # swap seq_len and h so that we can consider each head as (seq_len, d_k)
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)

        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)

        print(
            "\n\n\n\t\t\t\t\t MHA() called: After permutation \n\t\t\t\t\t\tquery: (batch, heads, seq_len, d_model) : ",
            query.shape,
        )

        print("\t\t\t\t\t\tkey: (batch, heads, seq_len, d_model) : ", key.shape)
        print("\t\t\t\t\t\tvalue: (batch, heads, seq_len, d_model) : ", value.shape)

        # (batch, h,seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # reverting shape permutation: (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k)
        x = x.permute(0, 2, 1, 3)

        # (batch, seq_len, h, d_k) -->(batch, seq_len, d_model)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k)
        x = self.wo(x)

        print("\tmhaReturnShape: (batch, seq_len, d_model) : ", x.shape)

        # (batch, seq_len, d_model
        return x


# # Create an instance of MultiHeadAttentionBlock
# MHA = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.5)

# # Create random input tensors
# batch_size = 8
# seq_len = 10
# d_model = 512

# # Create random tensors for query, key, and value
# # Shape: (batch_size, seq_len, d_model)
# q = torch.randn(batch_size, seq_len, d_model)
# k = torch.randn(batch_size, seq_len, d_model)
# v = torch.randn(batch_size, seq_len, d_model)

# # No mask for this example
# mask = None

# # Call the MultiHeadAttentionBlock
# mha = MHA(q, k, v, mask)

##################################################


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # x is input
        return x + self.dropout(sublayer(self.norm(x)))


###################################################


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block  # bve
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        print("\tencoderBlockReturnShape: (batch, seq_len, d_model) : ", x.shape)
        return x


###################################################
# for N encoder blocks


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers  # this would be encoderBlock
        self.norm = LayerNormalization()  # layerNorm at end

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            print(f"\n\nEncoder Block:{i+1} \n")
            x = layer(x, mask)
        x = self.norm(x)
        print("\n\tencoderReturnShape: (batch, seq_len, d_model) : ", x.shape)
        return x


######################################################3333
# # Example usage
# d_model = 512
# vocab_size = 1000
# seq_len = 35
# dropout = 0.5
# num_layers = 6
# num_heads = 8
# d_ff = 2048

# # Create instances
# input_embeddings = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
# positional_encoding = PositionalEncoding(d_model=d_model,
#                                          seq_len=seq_len,
#                                          dropout=dropout)
# feed_forward_block = FeedForwardBlock(d_model=d_model,
#                                       d_ff=d_ff,
#                                       dropout=dropout)
# self_attention_block = MultiHeadAttentionBlock(d_model=d_model,
#                                                h=num_heads,
#                                                dropout=dropout)

# encoder_blocks = nn.ModuleList([
#     EncoderBlock(self_attention_block, feed_forward_block, dropout)
#     for _ in range(num_layers)
# ])
# encoder = Encoder(layers=encoder_blocks)

# # Create example input
# batch_size = 1
# batch_of_sentences = torch.tensor(
#     [[5, 6, 7, 0, 0]])  # Shape: (batch_size, max_sentence_length)
# embedded_sentences = input_embeddings(batch_of_sentences)
# positional_encoded = positional_encoding(embedded_sentences)

# # Run through the encoder
# mask = None  # Assuming no mask for simplicity
# encoder_output = encoder(positional_encoded, mask)

# #################################################

# # output embeddings are same as input embeddings


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda t: self.self_attention_block(t, t, t, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda t: self.cross_attention_block(
                t, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        print("\tdecoderBlockReturnShape: (batch, seq_len, d_model) : ", x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for i, layer in enumerate(self.layers):
            print(f"\n\nDecoder Block:{i+1} \n")
            x = layer(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        print("\n\tdecoderReturnShape: (batch, seq_len, d_model) : ", x.shape)

        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = torch.log_softmax(self.proj(x), dim=-1)
        print("\n\tprojectionLayerReturnShape: (batch, seq_len, vocab_size) : ", x.shape)
        return x


####################################################

# d_model = 512
# vocab_size = 1000
# seq_len = 35
# dropout = 0.5
# num_layers = 6
# num_heads = 8
# d_ff = 2048

# input_embeddings = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
# positional_encoding = PositionalEncoding(d_model=d_model,
#                                          seq_len=seq_len,
#                                          dropout=dropout)
# feed_forward_block = FeedForwardBlock(d_model=d_model,
#                                       d_ff=d_ff,
#                                       dropout=dropout)
# self_attention_block = MultiHeadAttentionBlock(d_model=d_model,
#                                                h=num_heads,
#                                                dropout=dropout)

# projection_layer = ProjectionLayer(d_model, vocab_size)

# encoder_blocks = nn.ModuleList([
#     EncoderBlock(self_attention_block, feed_forward_block, dropout)
#     for _ in range(num_layers)
# ])
# encoder = Encoder(layers=encoder_blocks)

# # Decoder specific components
# cross_attention_block = MultiHeadAttentionBlock(d_model=d_model,
#                                                 h=num_heads,
#                                                 dropout=dropout)
# decoder_blocks = nn.ModuleList([
#     DecoderBlock(self_attention_block, cross_attention_block,
#                  feed_forward_block, dropout) for _ in range(num_layers)
# ])
# decoder = Decoder(layers=decoder_blocks)

# # Create example input
# batch_size = 1
# batch_of_sentences = torch.tensor(
#     [[5, 6, 7, 0, 0]])  # Shape: (batch_size, max_sentence_length)
# embedded_sentences = input_embeddings(batch_of_sentences)
# positional_encoded = positional_encoding(embedded_sentences)

# # Run through the encoder
# mask = None  # Assuming no mask for simplicity
# encoder_output = encoder(positional_encoded, mask)

# # Create example decoder input
# decoder_input = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (batch_size, seq_len)
# decoder_embedded = input_embeddings(decoder_input)
# decoder_positional_encoded = positional_encoding(decoder_embedded)

# # Run through the decoder
# tgt_mask = None  # Assuming no mask for simplicity
# decoder_output = decoder(decoder_positional_encoded, encoder_output, mask,
#                          tgt_mask)

# projecteded = projection_layer(decoder_output)

#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
#############################o##############################
# Transformer


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # three methods, one to encoder, one to decode and one to project
    # Not creating single forward method because we can reuse output of encoder and to also visualize the attention

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt

    def project(self, x):
        return self.projection_layer(x)


#####################################################################


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.5,
    d_ff: int = 2048,
) -> Transformer:
    """
    we need vocab size of src and tgt so get info about how many vectors to be created
    Keyword arguments:

    N: number of input layers i.e. number of enccoder blocks and number of decoder blocks
    h: # of heads
    """
    """strcuture will be same across all tasks"""
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # positional encoding layers
    # one encoding layer wold be enough
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # create encoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transfromer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # initilize parameter to make trainig faster so they dont just strat with random values
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


# transformer = build_transformer(
#     src_vocab_size=1000,
#     tgt_vocab_size=1000,
#     src_seq_len=100,
#     tgt_seq_len=100,
#     d_model=512,
#     N=6,
#     h=8,
#     dropout=0.5,
#     d_ff=2048,
# )
# print(transformer)
