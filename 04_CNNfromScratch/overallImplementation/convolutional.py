import numpy as np
from layer import Layer
from scipy import signal


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
        depth: # of kernels
        """
        self.input_depth, self.input_height, self.input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.output_shape = (
            self.depth,
            self.input_height - self.kernel_size + 1,
            self.input_width - self.kernel_size + 1,
        )
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.random.randn(*self.output_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        1. we need to compute two type of gradients
            i. gradients w.r.t trainable parameters i.e. kernels: dE/dKij and biases dE/dBi
            ii. derivative of Error w.r.t inputs dE/dXj

        """
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
