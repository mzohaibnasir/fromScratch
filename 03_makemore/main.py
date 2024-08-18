import torch
import matplotlib.pyplot as plt

with open("names.txt", "r") as f:
    words = f.read().split("\n")
    # print(words)


min_length = len(words[0])
smallest_word = words[0]
for i, w in enumerate(words):
    current_length = len(w)
    if min_length > current_length:
        min_length = current_length
        smallest_word = words[i]

# print(f"min_length: {min_length}, smallest_word: {smallest_word}")


# print(sorted(words, key=len, reverse=True)[:100])

# mantaining count
b = {}

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        # print(ch1, ch2)
        bigram = (ch1, ch2)
        b[bigram] = b[bigram] + 1 if bigram in b.keys() else 0

# print(b)

sorted_b = sorted(b.items(), key=lambda kv: kv[1], reverse=True)
ks = list(dict(sorted_b).keys())
# for i in ks[:15]:
#    print(i, b[i])

# print(b.keys())


# concurrency matrix
# alphabets= [set(i) for i in words[:2]]
alphabets = set("".join(words))
chars = ["."] + sorted(alphabets)  # indexes


stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
# print(stoi)
# print(itos)

# print(chars)

N = torch.zeros((len(chars), len(chars)), dtype=torch.int32)


for w in words:
    w = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(w, w[1:]):
        # concurrency_matrix[
        N[stoi[ch1], stoi[ch2]] += 1

# print(N[:5, :5])


# %matplotlib inline


itos = {i: s for s, i in stoi.items()}  # reverse dictionary

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(len(chars)):
    for j in range(len(chars)):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.imshow(N)


# Generator to manage deterministic randomness

g = torch.Generator().manual_seed(2147483647)

idx = 0
while True:
    p = N[idx].float()
    p /= p.sum()
    idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    print(itos[idx])
    if idx == 0:
        break
