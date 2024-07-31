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
        print("\tinputEmbeddingsReturnShape: (batch, seq_len, d_model) : ",
              x.shape)
        return x


# Example usage
input_embeddings = InputEmbeddings(d_model=512, vocab_size=1000)
# Create an example input tensor (batch size 1, sequence length 5, embedding dimension 20)
batch_of_sentences = torch.tensor(
    [[5, 6, 7, 0, 0]])  # Shape: (batch_size, max_sentence_length)
# print(batch_of_sentences.shape)

# Pass through the embedding layer
# The forward method is called automatically when you use the instance like a function.
embedded_sentences = input_embeddings(batch_of_sentences)
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
            1)  # (seq_len,1)  # pos in formula

        # create denominator
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

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
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # x:token and pe is positional encoding  # because we dont want to learn pe because these are fixed
        print("\tpositionalEncodingReturnShape: (batch, seq_len, d_model) : ",
              x.shape)

        #  :x.shape[1]:selecting just # of tokens because of input sequence length.
        return self.dropout(x)  # (batch, seq_len, d_model)


# Example usage
positional = PositionalEncoding(d_model=512, seq_len=35, dropout=0.5)
# print("positional:", positional)

# Create an example input tensor (batch size , sequence length , embedding dimension )

# print("\n\n\n input to PE", embedded_sentences.shape)

positional_encoded = positional(embedded_sentences)
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

        print("\tnormalizedReturnShape: (batch, seq_len, d_model) : ",
              normalized.shape)

        return normalized


ln = LayerNormalization()

# print("Before normalization:")
# print(positional_encoded)

normalized = ln(positional_encoded)
# print("After normalization:")
# print(normalized.shape)
# normalized  # (1, seq_len,d_model)

##########################################################################################

# feed forward network


class FeedForwardBlock(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 dropout: float = 0.5):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        # print("\tbeforeFeedForwardReturnShape: (batch, seq_len, d_model) : ",
        #       x.shape)

        self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        print("\tfeedForwardReturnShape: (batch, seq_len, d_model) : ",
              x.shape)

        return x


feedforwardblock = FeedForwardBlock()

feedforwarded = feedforwardblock(normalized)

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

        # weight matrices
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # output matrix :Wo (h*dv, d_model)
        self.wo = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # extract embedding length

        # (b,h,seq_len,dk)->b,h,seq_len,seq_len)
        attention_scores = torch.einsum("bhij,bhkj->bhik", query,
                                        key) / math.sqrt(d_k)

        if mask is not None:
            # replace all values with very small number so softwax will assign them 0 in output.
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (b,h,seq_len, seq_len) -> (b,h, seq_len, d_k
        attention_scoresV = torch.einsum("bhsj, bhsk->bhsk", attention_scores,
                                         value)
        print(
            "\t\tmultiHeadAttentionReturnShape: (batch,heads, seq_len, d_k) : ",
            attention_scoresV.shape,
        )

        return (attention_scoresV, attention_scores)

    def forward(self, q, k, v, mask):
        query = self.wq(q)
        value = self.wv(v)
        key = self.wk(k)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)

        # swap seq_len and h so that we can consider each head as (seq_len, d_k)
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)

        query = torch.permute(query, (0, 2, 1, 3))
        value = torch.permute(value, (0, 2, 1, 3))
        key = torch.permute(key, (0, 2, 1, 3))

        # (batch, h,seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask=None, dropout=None)

        # reverting shape permutation: (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k)
        x = torch.permute(x, (0, 2, 1, 3))

        # (batch, seq_len, h, d_k) -->(batch, seq_len, d_model)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k)
        x = self.wo(x)

        print("\tmhaReturnShape: (batch, seq_len, d_model) : ", x.shape)

        # (batch, seq_len, d_model
        return x


# Create an instance of MultiHeadAttentionBlock
MHA = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.5)

# Create random input tensors
batch_size = 8
seq_len = 10
d_model = 512

# Create random tensors for query, key, and value
# Shape: (batch_size, seq_len, d_model)
q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)
v = torch.randn(batch_size, seq_len, d_model)

# No mask for this example
mask = None

# Call the MultiHeadAttentionBlock
mha = MHA(q, k, v, mask)

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
        self.self_attention_block = self_attention_block  # bve
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


###################################################
# for N encoder blocks


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers  # this would be encoderBlock
        self.norm = LayerNormalization()  # layerNorm at end

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


#################################################
