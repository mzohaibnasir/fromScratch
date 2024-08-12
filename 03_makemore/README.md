# Makemore
makemore makes more of the things you give him
an autoregressive character level language-model, with a iwde choice of models from bigrams, all the way to transformer(exactly as seen in GPT).
We can feed it a database of names and it  will generate cool baby name ideas that all sound name-like, but are not already existing names.

Character level LM means it is treating every line in input file i.e. each name in input file as an example and wothin each example its treating each example as a sequence of individual characters. so it generates one character at a time.


Current language models neural nets implemented:
1. Bigram(one character simply predicts a next one with a lookup table of counts)
2. BoW
3. MLP
4. RNN
5. GRU
6. Transformer