import cv2
import torch
import torchvision
from torch import tensor
import torchvision.datasets as datasets

# Download and load the MNIST dataset
mnist = datasets.MNIST(root="./data", train=False, download=True)

# Print the length of the dataset
print(len(mnist))

# Access the first sample (image and label) in the dataset
image, label = mnist[0]

# Print the shape of the image (as a PIL image, it doesn't have a shape like a NumPy array)
print("Label:", label)

# Convert the image to a NumPy array
# "image_np = torchvision.transforms.ToTensor()(image).numpy()
image_np = torch.tensor(image)

# Convert from (1, H, W) to (H, W) since MNIST images are grayscale
image_np = image_np[0]
print(image_np.shape)
# Display the image using OpenCV
# cv2.imshow("MNIST Image", image_np)

# Wait for a key press and close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
print("matrix multiplication")
torch.manual_seed(1)
weights = torch.randn(784, 10)
bias = torch.zeros(10)
