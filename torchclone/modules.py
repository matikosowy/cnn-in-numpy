import numpy as np
from typing import Union, Tuple

class Conv2d:
    """
    Convolutional layer for 2D matrices (NCHW format)
    N - batch size
    C - number of channels
    H - height
    W - width

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
        self.training = True

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
        
        self.cache = {}

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
            
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the convolutional layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        np.ndarray
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape

        # Cache input for backward pass
        self.cache['input'] = x

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

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the convolutional layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        input_data = self.cache['input']
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
                self.weight_grad += np.einsum(
                    'bchw,bo->ochw',
                    input_slice,
                    grad_output[:, :, i, j]
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
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride})"


class ConvTranspose2d:
    """
    Transposed convolutional layer for 2D matrices (NCHW format)
    N - batch size
    C - number of channels
    H - height
    W - width

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
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[str, int] = 'same',
                 padding_mode: str = 'zeros',
                 bias: bool = True,
                 init: str = 'he'
                 ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding_mode = padding_mode

        # Initialize weights and bias
        self.weight = np.random.randn(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1])

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
        
        self.cache = {}
        
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
        else:
            return np.pad(
                x,
                ((0, 0), (0, 0),
                 (self.padding[0], self.padding[0]),
                 (self.padding[1], self.padding[1])),
                mode='reflect'
            )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Transposed convolutional layer for 2D matrices (NCHW format)
        N - batch size
        C - number of channels
        H - height
        W - width

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
        pass
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the transposed convolutional layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        pass
    
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)
    
    def __str__(self):
        return f"ConvTranspose2d({self.in_channels}, {self.out_channels}, {self.kernel_size})"


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

        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the linear layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor of shape (batch_size, in_features)

        Returns:
        --------
        np.ndarray
            Output tensor of shape (batch_size, out_features)
        """
        self.cache['input'] = x
        output = x @ self.weight.T

        if self.bias is not None:
            output += self.bias

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the linear layer.

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
        input_data = self.cache['input']

        # Reset gradients
        self.weight_grad.fill(0.0)
        if self.bias is not None:
            self.bias_grad.fill(0.0)

        # Compute weight gradients
        self.weight_grad = grad_output.T @ input_data

        # Compute bias gradients
        if self.bias is not None:
            self.bias_grad = np.sum(grad_output, axis=0)

        # Compute input gradients
        grad_input = grad_output @ self.weight

        return grad_input
        
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class BatchNorm2d:
    """
    Batch normalization layer for 2D matrices (NCHW format)
    N - batch size
    C - number of channels
    H - height
    W - width

    Parameters:
    -----------
    n_features : int
        Number of features in the input tensor
    eps : float
        Epsilon value for numerical stability
    momentum : float
        Momentum for running mean and variance
    affine : bool
        Whether to apply an affine transformation
    training : bool
        Whether the model is in training mode

    Returns:
    --------
    np.ndarray
        Output tensor of shape (batch_size, n_features, height, width)
    """
    def __init__(self, n_features: int,
                 eps: float=1e-5,
                 momentum: float=0.1,
                 affine: bool=True,
                 training: bool=True
    ):
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
            self.weight_grad = np.zeros(n_features)
            self.bias_grad = np.zeros(n_features)
        else:
            self.weight = None
            self.bias = None
            self.weight_grad = None
            self.bias_grad = None

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the batch normalization layer.

        Parameters:
        ___________
        x : np.ndarray
            Input tensor of shape (batch_size, n_features, height, width)

        Returns:
        ________
        np.ndarray
            Output tensor of shape (batch_size, n_features, height, width)
        """
        self.cache['input'] = x

        if self.training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, self.n_features, 1, 1)
            var = self.running_var.reshape(1, self.n_features, 1, 1)

        # Normalize
        x_centered = x - mean
        std = np.sqrt(var + self.eps)
        x_normalized = x_centered / std

        self.cache.update({
            'mean': mean,
            'var': var,
            'std': std,
            'x_centered': x_centered,
            'x_normalized': x_normalized
        })

        if self.affine:
            output = (self.weight.reshape(1, self.n_features, 1, 1) * x_normalized
                      + self.bias.reshape(1, self.n_features, 1, 1))
        else:
            output = x_normalized

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for batch normalization layer.

        Parameters:
        ___________
            grad_output: Gradient of the loss with respect to the output

        Returns:
        ________
        np.ndarray
            Gradient of the loss with respect to the input
        """
        x = self.cache['input']
        mean = self.cache['mean']
        var = self.cache['var']
        std = self.cache['std']
        x_centered = self.cache['x_centered']
        x_norm = self.cache['x_normalized']

        N = x.shape[0] * x.shape[2] * x.shape[3]

        if self.affine:
            # Gradients with respect to weight and bias
            self.weight_grad = np.sum(grad_output * x_norm, axis=(0, 2, 3))
            self.bias_grad = np.sum(grad_output, axis=(0, 2, 3))

            # Adjust grad_output for affine
            grad_output = grad_output * self.weight.reshape(1, -1, 1, 1)

        # Gradients with respect to input
        grad_normalized = grad_output
        grad_var = np.sum(grad_normalized * x_centered * -0.5 * std ** (-3), axis=(0, 2, 3), keepdims=True)
        grad_mean = np.sum(grad_normalized * -1 / std, axis=(0, 2, 3), keepdims=True)

        grad_input = (grad_normalized / std +
                      2 * x_centered * grad_var / N +
                      grad_mean / N)

        return grad_input

    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return f"BatchNorm2d({self.n_features})"


class BathNorm1d:
    """
    Batch normalization layer for 1D tensors (NC format)
    N - batch size
    C - number of channels

    Parameters:
    -----------
    n_features : int
        Number of features in the input tensor
    eps : float
        Epsilon value for numerical stability
    momentum : float
        Momentum for running mean and variance
    affine : bool
        Whether to apply an affine transformation
    training : bool
        Whether the model is in training mode

    Returns:
    --------
    np.ndarray
        Output tensor of shape (batch_size, n_features)
    """
    def __init__(self, n_features: int,
                 eps: float=1e-5,
                 momentum: float=0.1,
                 affine: bool=True,
                 training: bool=True
    ):
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
            self.weight_grad = np.zeros(n_features)
            self.bias_grad = np.zeros(n_features)
        else:
            self.weight = None
            self.bias = None
            self.weight_grad = None
            self.bias_grad = None

        self.cache = {}
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the batch normalization layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor of shape (batch_size, n_features)

        Returns:
        --------
        np.ndarray
            Output tensor of shape (batch_size, n_features)
        """
        self.cache['input'] = x

        if self.training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, self.n_features)
            var = self.running_var.reshape(1, self.n_features)

        # Normalize
        x_centered = x - mean
        std = np.sqrt(var + self.eps)
        x_normalized = x_centered / std

        self.cache.update({
            'mean': mean,
            'var': var,
            'std': std,
            'x_centered': x_centered,
            'x_normalized': x_normalized
        })

        if self.affine:
            output = self.weight * x_normalized + self.bias
        else:
            output = x_normalized

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for batch normalization layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        x = self.cache['input']
        mean = self.cache['mean']
        var = self.cache['var']
        std = self.cache['std']
        x_centered = self.cache['x_centered']
        x_norm = self.cache['x_normalized']

        N = x.shape[0]

        if self.affine:
            # Gradients with respect to weight and bias
            self.weight_grad = np.sum(grad_output * x_norm, axis=0)
            self.bias_grad = np.sum(grad_output, axis=0)

            # Adjust grad_output for affine
            grad_output = grad_output * self.weight

        # Gradients with respect to input
        grad_normalized = grad_output
        grad_var = np.sum(grad_normalized * x_centered * -0.5 * std ** (-3), axis=0, keepdims=True)
        grad_mean = np.sum(grad_normalized * -1 / std, axis=0, keepdims=True)

        grad_input = (grad_normalized / std +
                      2 * x_centered * grad_var / N +
                      grad_mean / N)

        return grad_input

    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return f"BatchNorm1d({self.n_features})"


class MaxPool2d:
    """
    Max pooling layer for 2D matrices (NCHW format)
    N - batch size
    C - number of channels
    H - height
    W - width

    Parameters:
    -----------
    kernel_size : Union[int, Tuple[int, int]]
        Size of the pooling kernel
    stride : Union[int, Tuple[int, int]]
        Stride of the pooling operation

    Returns:
    --------
    np.ndarray
        Output tensor of shape (batch_size, in_channels, out_height, out_width)
    """
    def __init__(self, kernel_size: Union[int, tuple] = (2, 2),
                 stride: Union[int, tuple] = (2, 2)
        ):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the max pooling layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        np.ndarray
            Output tensor of shape (batch_size, in_channels, out_height, out_width)
        """
        self.cache['input'] = x
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

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for max pooling layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        input_data = self.cache['input']
        batch_size, channels, height, width = input_data.shape
        _, _, out_height, out_width = grad_output.shape

        grad_input = np.zeros_like(input_data)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # Get input slice
                input_slice = input_data[:, :, h_start:h_end, w_start:w_end]

                # Create mask of where maximum values are
                mask = (input_slice == np.max(input_slice, axis=(2, 3))
                        .reshape(batch_size, channels, 1, 1))

                # Distribute gradient
                grad_input[:, :, h_start:h_end, w_start:w_end] += \
                    mask * grad_output[:, :, i, j].reshape(batch_size, channels, 1, 1)

        return grad_input

    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return f"MaxPool2d({self.kernel_size}, {self.stride})"


class Dropout:
    """
    Dropout layer

    Parameters:
    -----------
    p : float
        Probability of dropping out a neuron
    training : bool
        Whether the model is in training mode
    """
    def __init__(self, p: float=0.5, training: bool=True):
        self.p = p
        self.training = training
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the dropout layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Output tensor
        """
        if self.training:
            mask = np.random.rand(*x.shape) > self.p
            self.cache['mask'] = mask
            return x * mask
        else:
            return x
        
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for dropout layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        mask = self.cache['mask']
        return grad_output * mask

    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return f"Dropout({self.p})"


class ReLU:
    """Rectified Linear Unit (ReLU) activation function"""
    def __init__(self):
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the ReLU activation function.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Output tensor
        """
        self.cache['input'] = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for ReLU activation function.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        input_data = self.cache['input']
        return grad_output * (input_data > 0)
    
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return "ReLU()"


class Softmax:
    """
    Softmax activation function

    Parameters:
    -----------
    axis : int
        Axis along which to compute the softmax
    """
    def __init__(self, axis: int=-1):
        self.axis = axis
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the softmax activation function.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Output tensor
        """
        # For numerical stability
        shifted_x = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted_x)
        softmax_x = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        self.cache['output'] = softmax_x
        return softmax_x

    def backward(self, grad_output: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for softmax activation function.

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
        softmax_x = self.cache['output']
        return grad_output * softmax_x * (1 - softmax_x)
    
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return "Softmax()"
    

class Sigmoid:
    """
    Sigmoid activation function
    """
    def __init__(self):
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the sigmoid activation function.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Output tensor
        """
        sigmoid_x = 1 / (1 + np.exp(-x))
        self.cache['output'] = sigmoid_x
        return sigmoid_x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for sigmoid activation function.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        sigmoid_x = self.cache['output']
        return grad_output * sigmoid_x * (1 - sigmoid_x)
    
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return "Sigmoid()"
    

class Tanh:
    """Hyperbolic tangent activation function"""
    def __init__(self):
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the tanh activation function.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Output tensor
        """
        tanh_x = np.tanh(x)
        self.cache['output'] = tanh_x
        return tanh_x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for tanh activation function.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        tanh_x = self.cache['output']
        return grad_output * (1 - tanh_x ** 2)
    
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)
    
    def __str__(self):
        return "Tanh()"
    

class LeakyReLU:
    """Leaky Rectified Linear Unit (Leaky ReLU) activation function
    
    Parameters:
    -----------
    slope : float
        Slope of the negative part of the function
    """
    def __init__(self, slope: float=0.01):
        self.slope = slope
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the Leaky ReLU activation function.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Output tensor
        """
        self.cache['input'] = x
        return np.where(x > 0, x, self.slope * x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for Leaky ReLU activation function.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        input_data = self.cache['input']
        return np.where(input_data > 0, grad_output, self.slope * grad_output)
    
    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)
    
    def __str__(self):
        return f"LeakyReLU({self.alpha})"


class Flatten:
    """
    Flatten layer to flatten the input tensor
    """
    def __init__(self):
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the flatten layer.

        Parameters:
        -----------
        x : np.ndarray
            Input tensor

        Returns:
        --------
        np.ndarray
            Flattened tensor
        """
        self.cache['input_shape'] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for flatten layer.

        Parameters:
        -----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of this layer

        Returns:
        --------
        np.ndarray
            Gradient of the loss with respect to the input
        """
        return grad_output.reshape(self.cache['input_shape'])

    def __call__(self, x):
        """Shortcut to the forward method"""
        return self.forward(x)

    def __str__(self):
        return "Flatten()"