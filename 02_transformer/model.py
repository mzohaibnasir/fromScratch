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

        """
        
        formula :`PE(pos,2i) = cos(pos/(10000)**2i/dmodel) for i=1,3,5, ...and `PE(pos,2i) = sin(pos/(10000)**2i/dmodel) for i=2,4,6, ...and `
        
        """

        #  Create a model of shape (seq_len , d_model)

        pe = torch.zeros(seq_len, d_model)
        #  create a vector of shape(seq_len,1) to represent position of word in sequence

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len,1)  # pos in formula
        # create denominator of formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # now we need to add batch dimension to these sentences so we can apply it to whole sentences, so to all the batch of sentence, because weill have batch of sentences.
        # adding batch dim
        pe = pe.unsqueeze(0)  # (1, seq_len,d_model)

        # register this tensor in buffer of module  .. it is done for the tensor that you want to keep inside the module, not as a lerarned parameter but you want it to be saved when you save the file of the model
        # you should register it as a buffer. this way the tensor would be saved in file along with state of model
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        we need to add positional encoding to every token/word inside sequence/sentence
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)  # x:token and pe is positional encoding  # because we dont want to learn pe because these are fixed
        return self.dropout(x)
