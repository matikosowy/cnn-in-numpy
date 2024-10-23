import numpy as np
from typing import Union, Tuple, Optional

# todo:
# - implement backward passes for all layers

class Conv2d:
    """
    Convolutional layer for 2D matrices (NCHW format)

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : Union[int, Tuple[int, int]]
        Size of the convolutional kernel
    stride : Union[int, Tuple[int, int]]
        Stride of the convolution
    padding : Union[str, int]
        'valid', 'same', or number of padding pixels
    padding_mode : str
        'zeros' or 'reflect'
    bias : bool
        Whether to include a bias term
    init : str
        Weight initialization method ('he' or 'normal')
    """

    def __init__(self, in_channels: int, 
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = (3,3),
                 stride: Union[int, Tuple[int, int]] = (1,1),
                 padding: Union[str, int] = 'same',
                 padding_mode: str='zeros',
                 bias: bool=True,
                 init: str='he'
        ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding_mode = padding_mode

        # Initialize weights and bias
        self.weight = np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

        if bias:
            self.bias = np.zeros(out_channels)
        else:
            self.bias = None

        if init == 'he':
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
            self.weight *= np.sqrt(2.0 / fan_in)
        elif init == 'normal':
            self.weight *= 0.01
        else:
            raise ValueError('Invalid initialization method! Use "he" or "normal".')

        # Set padding
        if padding_mode not in ['zeros', 'reflect']:
            raise ValueError('Invalid padding mode! Use "zeros" or "reflect".')

        if isinstance(padding, str):
            if padding == 'same':
                self.padding = (
                    (self.kernel_size[0] - 1) // 2,
                    (self.kernel_size[1] - 1) // 2
                )
            elif padding == 'valid':
                self.padding = (0, 0)
            else:
                raise ValueError('Invalid padding! Use "same", "valid", or an integer.')
        else:
            self.padding = (padding, padding)
            
        # Initialize gradients
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias) if bias else None
        
        self.input_shape = None

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Apply padding to input tensor"""
        if self.padding == (0, 0):
            return x

        if self.padding_mode == 'zeros':
            return np.pad(
                x,
                ((0, 0), (0, 0),
                 (self.padding[0], self.padding[0]),
                 (self.padding[1], self.padding[1]))
            )
        else:  # 'reflect'
            return np.pad(
                x,
                ((0, 0), (0, 0),
                 (self.padding[0], self.padding[0]),
                 (self.padding[1], self.padding[1])),
                mode='reflect'
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the convolutional layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        np.ndarray
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        self.input_shape = x.shape
        batch_size, in_channels, height, width = x.shape

        # Apply padding
        x_padded = self._pad_input(x)
        pad_height, pad_width = x_padded.shape[2:]

        # Calculate output dimensions
        out_height = ((pad_height - self.kernel_size[0]) // self.stride[0]) + 1
        out_width = ((pad_width - self.kernel_size[1]) // self.stride[1]) + 1

        if out_height <= 0 or out_width <= 0:
            raise ValueError("Output dimensions must be positive. Check kernel size and stride.")

        # Initialize output tensor
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.einsum('nchw,ochw->no', x_slice, self.weight)

        if self.bias is not None:
            output += self.bias[None, :, None, None]

        return output

    def backward(self, grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        Backward pass of the convolutional layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer
        input_data : np.ndarray
            The input that was used in the forward pass

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        batch_size, _, out_height, out_width = grad_output.shape

        # Reset gradients
        self.weight_grad.fill(0.0)
        if self.bias is not None:
            self.bias_grad.fill(0.0)

        # Pad input if necessary
        input_padded = self._pad_input(input_data)

        # Initialize gradient with respect to input
        grad_input_padded = np.zeros_like(input_padded)

        # Compute gradients
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                input_slice = input_padded[:, :, h_start:h_end, w_start:w_end]

                # Compute weight gradients
                for b in range(batch_size):
                    self.weight_grad += np.einsum(
                        'chw,o->ochw',
                        input_slice[b],
                        grad_output[b, :, i, j]
                    )

                # Compute input gradients
                grad_input_padded[:, :, h_start:h_end, w_start:w_end] += np.einsum(
                    'ochw,bo->bchw',
                    self.weight,
                    grad_output[:, :, i, j]
                )

        # Compute bias gradients
        if self.bias is not None:
            self.bias_grad = np.sum(grad_output, axis=(0, 2, 3))

        # Remove padding from gradient if necessary
        if self.padding != (0, 0):
            grad_input = grad_input_padded[:, :,
                         self.padding[0]:-self.padding[0],
                         self.padding[1]:-self.padding[1]]
        else:
            grad_input = grad_input_padded

        return grad_input

    def __call__(self, x):
        return self.forward(x)


class Linear:
    """
    Fully connected (dense) layer

    Parameters:
    -----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        Whether to include a bias term
    init : str
        Weight initialization method ('he' or 'normal')
    """
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool=True,
                 init: str='normal'
    ):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias
        self.weight = np.random.randn(out_features, in_features)
        
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None

        if init == 'he':
            self.weight *= np.sqrt(2.0 / in_features)
        elif init == 'normal':
            self.weight *= 0.01
        else:
            raise ValueError('Invalid initialization method! Use "he" or "normal".')

        # Initialize gradients
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias) if bias else None

        self.input_shape = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the linear layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor of shape (batch_size, in_features)

        Returns:
        --------
        np.ndarray
            Output tensor of shape (batch_size, out_features)
        """
        self.input_shape = x.shape
        output = x @ self.weight.T

        if self.bias is not None:
            output += self.bias

        return output

    def backward(self, grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        Backward pass of the linear layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer
        input_data : np.ndarray
            The input that was used in the forward pass

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        # Reset gradients
        self.weight_grad.fill(0.0)
        if self.bias is not None:
            self.bias_grad.fill(0.0)

        # Compute weight gradients
        self.weight_grad = np.einsum('bi,bj->ij', grad_output, input_data)

        # Compute bias gradients
        if self.bias is not None:
            self.bias_grad = np.sum(grad_output, axis=0)

        # Compute input gradients
        grad_input = grad_output @ self.weight

        return grad_input
        
    def __call__(self, x):
        return self.forward(x)


class BatchNorm2d:
    """
    Batch normalization layer for 2D matrices (NCHW format)

    Parameters:

    """
    def __init__(self, n_features: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True,
                 training: bool=True):
        self.n_features = n_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.training = training

        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)

        if affine:
            self.weight = np.ones(n_features)
            self.bias = np.zeros(n_features)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the batch normalization layer.

        Args:
        x (np.ndarray): Input tensor.

        Returns:
        np.ndarray: Output tensor.
        """
        if self.training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, self.n_features, 1, 1)
            var = self.running_var.reshape(1, self.n_features, 1, 1)

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            output = (self.weight.reshape(1, self.n_features, 1, 1) * x_hat
                      + self.bias.reshape(1, self.n_features, 1, 1))
        else:
            output = x_hat

        return output

    def __call__(self, x):
        return self.forward(x)


class MaxPool2d:
    def __init__(self, kernel_size: int or tuple = (2, 2), stride: int or tuple = (2, 2)):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the max pooling layer.

        Args:
        x (np.ndarray): Input tensor.

        Returns:
        np.ndarray: Output tensor.
        """
        batch_size, in_channels, height, width = x.shape

        # Calculate output dimensions
        height_out = (height - self.kernel_size[0]) // self.stride[0] + 1
        width_out = (width - self.kernel_size[1]) // self.stride[1] + 1

        if height_out <= 0 or width_out <= 0:
            raise ValueError("Output dimensions must be positive. Check kernel size and stride.")

        output = np.zeros((batch_size, in_channels, height_out, width_out))

        for i in range(height_out):
            for j in range(width_out):
                x_slice = x[:, :,
                          i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                          j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]

                output[:, :, i, j] = np.max(x_slice, axis=(2, 3))

        return output

    def __call__(self, x):
        return self.forward(x)


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function
    """
    def __init__(self):
        self.f_relu = np.vectorize(lambda x: np.maximum(0, x))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the ReLU activation function.

        Args:
        x (np.ndarray): Input tensor.

        Returns:
        np.ndarray: Output tensor.
        """
        return self.f_relu(x)
    
    def __call__(self, x):
        return self.forward(x)


class Softmax:
    """
    Softmax activation function

    Parameters:
    axis (int): Axis across which to apply the softmax function.
    """
    def __init__(self, axis: int=-1):
        self.f_softmax = np.vectorize(lambda x: np.exp(x) / np.sum(np.exp(x), axis=axis))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the softmax activation function.

        Args:
        x (np.ndarray): Input tensor.

        Returns:
        np.ndarray: Output tensor.
        """
        return self.f_softmax(x)
    
    def __call__(self, x):
        return self.forward(x)
    

class Sigmoid:
    """
    Sigmoid activation function
    """
    def __init__(self):
        self.f_sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the sigmoid activation function.

        Args:
        x (np.ndarray): Input tensor.

        Returns:
        np.ndarray: Output tensor.
        """
        return self.f_sigmoid(x)
    
    def __call__(self, x):
        return self.forward(x)


class Flatten:
    """
    Flatten layer to flatten the input tensor
    """
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the flatten layer.

        Args:
        x (np.ndarray): Input tensor.

        Returns:
        np.ndarray: Output tensor.
        """
        return x.reshape(x.shape[0], -1)

    def __call__(self, x):
        return self.forward(x)