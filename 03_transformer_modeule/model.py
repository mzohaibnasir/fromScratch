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
        print("\tSampleInputShape: (batch, seq_len) : ", x.shape)
        x = self.embedding(x)
        print("\tInputEmbeddingsReturnShape: (batch, seq_len, d_model) : ",
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
        print("\tPositionalEncodingReturnShape: (batch, seq_len, d_model) : ",
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
