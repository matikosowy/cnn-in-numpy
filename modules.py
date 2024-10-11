import numpy as np

# todo:
# - implement backward passes for all layers
# - implement optimizer classes (SGD, Adam)
# - implement loss functions (CrossEntropyLoss, MSELoss)
# - implement training loop

class Conv2d:
    """
    Convolutional layer for 2D matrices (NCHW format)

    Parameters:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (tuple or int): Size of the kernel (can be a tuple or int).
    stride (tuple or int): Stride of the convolution (can be a tuple or int).
    padding (str or int): Padding type or number of padding.
    padding_mode (str): Padding mode ('zeros' or 'reflect').
    bias (bool): Whether to use bias or not.
    init (str): Initialization method for the weights ('he' or 'normal').
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int or tuple=(3,3), stride: int or tuple=(2, 2), padding: str='same',
                 padding_mode: str='zeros', bias: bool=True, init: str='he'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding_mode = padding_mode

        self.weight = np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

        if bias:
            self.bias = np.zeros(out_channels)
        else:
            self.bias = None

        if init == 'he':
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
            self.weight *= np.sqrt(2.0 / fan_in)
        elif init != 'normal':
            raise ValueError('Invalid initialization method! Use "he" or "normal".')

        if padding_mode not in ['zeros', 'reflect']:
            raise ValueError('Invalid padding mode! Use "zeros" or "reflect".')

        if padding == 'same':
            self.padding = (
                (self.kernel_size[0] - 1) // 2,
                (self.kernel_size[1] - 1) // 2
            )
        elif padding == 'valid':
            self.padding = (0, 0)
        else:
            self.padding = padding

    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the convolutional layer.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.

        """
        batch_size, in_channels, height, width = x.shape

        if self.padding_mode == 'zeros':
            x = np.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]),
                           (self.padding[1], self.padding[1])))
        elif self.padding_mode == 'reflect':
            x = np.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]),
                           (self.padding[1], self.padding[1])), mode='reflect')

        height_out = np.floor(((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1
                              ).astype(int)
        width_out = np.floor(((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1
                             ).astype(int)

        if height_out <= 0 or width_out <= 0:
            raise ValueError("Output dimensions must be positive. Check kernel size and stride.")

        output = np.zeros((batch_size, self.out_channels, height_out, width_out))

        for i in range(height_out):
            for j in range(width_out):
                x_slice = x[:, :,
                          i * self.stride[0]:i * self.stride[0] + self.kernel_size[0],
                          j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]]

                output[:, :, i, j] = np.einsum('nchw,ochw->no', x_slice, self.weight)

        if self.bias is not None:
            output += self.bias[None, :, None, None]

        return output

    def __call__(self, x):
        return self.forward(x)


class Linear:
    """
    Fully connected layer (dense)

    Parameters:
    in_features (int): Number of input features.
    out_features (int): Number of output features.
    bias (bool): Whether to use bias or not.
    init (str): Initialization method for the weights ('he' or 'normal').
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=True, init: str='normal'):
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = np.random.randn(out_features, in_features)
        
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None
        
        if init == 'he':
            self.weight *= np.sqrt(2.0 / in_features)
        elif init != 'normal':
            raise ValueError('Invalid initialization method! Use "he" or "normal".')
        
    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the fully connected layer.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
        """
        return x @ self.weight.T + self.bias
        
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

    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the batch normalization layer.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
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

    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the max pooling layer.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
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
        
    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the ReLU activation function.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
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
        
    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the softmax activation function.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
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

    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the sigmoid activation function.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
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

    def forward(self, x: np.array) -> np.array:
        """
        Forward pass of the flatten layer.

        Args:
        x (np.array): Input tensor.

        Returns:
        np.array: Output tensor.
        """
        return x.reshape(x.shape[0], -1)

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':
    conv = Conv2d(3, 64, (4,4), 2, 'same', padding_mode='reflect')
    norm = BatchNorm2d(64)
    relu = ReLU()
    pool = MaxPool2d()
    flatten = Flatten()
    fc = Linear(64*4*4, 10)
    softmax = Softmax()

    x = np.random.randn(4, 3, 16, 16)
    print("Input shape:", x.shape)
    x = conv(x)
    print("Conv shape:", x.shape)
    x = norm(x)
    print("Norm shape:", x.shape)
    x = relu(x)
    print("ReLU shape:", x.shape)
    x = pool(x)
    print("Pool shape:", x.shape)
    x = flatten(x)
    print("Flatten shape:", x.shape)
    x = fc(x)
    print("FC shape:", x.shape)
    x = softmax(x)
    print("Softmax shape:", x.shape)
