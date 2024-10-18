# gpu enabled 
from tqdm import tqdm
import torch
import torch.nn.functional as F

# Move computations to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
words = open('names.txt', 'r').read().split("\n")
# Prepare data

chars =sorted(list(set("".join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}

stoi['.'] = 0
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# Convert to tensors and move to the device
xs = torch.tensor(xs, device=device)
ys = torch.tensor(ys, device=device)
nums = xs.nelement()
print(f"Number of elements: {nums}")

# Initialize weights with the proper device placement and requires_grad set to True
g = torch.Generator(device=device).manual_seed(2147783647)
W = torch.randn((27, 27), generator=g, requires_grad=True, device=device)

# Gradient descent loop
for k in tqdm(range(900000)):
    # Forward pass
    xenc = F.one_hot(xs, num_classes=27).float().to(device)  # Input to the network
    logits = xenc @ W  # Predicted log counts
    counts = logits.exp()  # Convert logits to counts
    probs = counts / counts.sum(dim=1, keepdim=True)  # Softmax probabilities

    # Compute the loss with regularization
    loss = -probs[torch.arange(nums, device=device), ys].log().mean() + 0.01 * (W**2).mean()
    if k % 10000 == 0:  # Print every 1000 iterations for tracking
        print(f"{k=:05d};    {loss.item()=}")

    # Backward pass
    W.grad = None  # Reset gradients
    loss.backward()

    # Update weights with gradient descent (in-place update)
    W.data.add_(-0.01 * W.grad)

#################################



g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0

    while True:


        # BEFORE
        # p = P[ix]

        # now
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float() # input to the network: one hot encoding
        logits = xenc @ W # predict log_counts
        counts = logits.exp() # counts equivalent to N
        p= counts / counts.sum(1, keepdim=True)  # probabilities for next character
        # print(p.shape)




        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0:
            break
    print(''.join(out))