import numpy as np

class Conv2d:
    """
    Convolutional layer for 2D matrices (NCHW format)
    
    Parameters:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (int): Size of the kernel (square).
    stride (int): Stride of the convolution.
    padding (str or int): Padding type or number of padding.
    padding_mode (str): Padding mode ('zeros' or 'reflect').
    bias (bool): Whether to use bias or not.
    init (str): Initialization method for the weights ('he' or 'normal').  
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int=3, stride: int=2, padding: str='same', 
                 padding_mode: str='zeros', bias: bool=True, init: str='he'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        
        if bias:
            self.bias = np.zeros(out_channels)
        else:
            self.bias = None
            
        if init == 'he':
            fan_in = in_channels * kernel_size * kernel_size
            self.weight *= np.sqrt(2.0 / fan_in)
        elif init != 'normal':
            raise ValueError('Invalid initialization method! Use "he" or "normal".')
        
        if padding_mode not in ['zeros', 'reflect']:
            raise ValueError('Invalid padding mode! Use "zeros" or "reflect".')
        
        if padding == 'same':
            self.padding = (kernel_size - 1) // 2
        elif padding == 'valid':
            self.padding = 0
        else:
            self.padding = padding
               
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        
        if self.padding_mode == 'zeros':
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        elif self.padding_mode == 'reflect':
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
           
        height_out = np.floor(((height + 2 * self.padding - self.kernel_size) / self.stride) + 1).astype(int)
        width_out = np.floor(((width + 2 * self.padding - self.kernel_size) / self.stride) + 1).astype(int)
        
        output = np.zeros((batch_size, self.out_channels, height_out, width_out))
        
        for n in range(batch_size):
            for c in range(self.out_channels):
                for h in range(height_out):
                    for w in range(width_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        output[n, c, h, w] = np.sum(x[n, :, h_start:h_end, w_start:w_end] * self.weight[c])
                        if self.bias is not None:
                            output[n, c, h, w] += self.bias[c]
        
        return output
            
          
    def __call__(self, x):
        return self.forward(x)
        

class Linear:
    def __init__(self, in_features, out_features, bias=True, init='normal'):
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
        
    def forward(self, x):
        pass
        
    def __call__(self, x):
        return self.forward(x)


class ReLU:
    def __init__(self):
        self.f_relu = np.vectorize(lambda x: max(0, x))
        
    def forward(self, x):
        return self.f_relu(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    
class MaxPool2d:
    pass


class BatchNorm2d:
    pass
        

class Softmax:
    def __init__(self):
        self.f_softmax = np.vectorize(lambda x: np.exp(x) / np.sum(np.exp(x)))
        
    def forward(self, x):
        return self.f_softmax(x)
    
    def __call__(self, x):
        return self.forward(x)
    

class Sigmoid:
    def __init__(self):
        self.f_sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        
    def forward(self, x):
        return self.f_sigmoid(x)
    
    def __call__(self, x):
        return self.forward(x)
    
        
if __name__ == '__main__':
    conv = Conv2d(3, 64, 4, 2, 'same', padding_mode='reflect')
    x = np.random.randn(1, 3, 32, 32)
    y = conv(x)
    print(y.shape)