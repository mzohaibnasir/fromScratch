# first we;ll be building input embeddings
# allows to convert token into embedding of dim 1x52  : token -> input ID(position in vocab) ->embedding


import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """

        Args:
            d_model (int): dim of vector
            vocab_size (int): # of words in vocab
        """

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


#######################################################################################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
                Since our model contains no recurrence and no convolution, in order for the model to make use of the
        order of the sequence, we must inject some information about the relative or absolute position of the
        tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the
        bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
        as the embeddings, so that the two can be summed. There are many choices of positional encodings,
        learned and fixed [9].
        In this work, we use sine and cosine functions of different frequencies:
            `PE(pos,2i) = sin(pos/(10000)**2i/dmodel)`
            `PE(pos,2i+1) = cos(pos/(10000)**2i/dmodel)`
        where pos is the position and i is the dimension. That is, each dimension of the positional encoding
        corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We
        chose this function because we hypothesized it would allow the model to easily learn to attend by
        relative positions, since for any fixed offset k, P E(pos+k) can be represented as a linear function of
        PE(pos).

        Keyword arguments:
        dropout -- to make model less overfit
        seq_len -- Specifies the maximum sequence length that the model can handle. This helps determine the scale and range of the positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # positional encodeing shape: seq_len X d_model i.e. each token will be represented (1*d_model) vector
        #  Create a model of shape (seq_len , d_model)

        pe = torch.zeros(seq_len, d_model)
