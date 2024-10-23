# todo:
# - implement training loop

from modules import *

if __name__ == '__main__':
    # Test Conv2d
    conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding='same')
    x = np.random.randn(32, 3, 28, 28)  # batch_size=32, channels=3, height=28, width=28

    # Forward pass
    output = conv(x)
    print("Conv2d output shape:", output.shape)

    # Backward pass
    grad_output = np.random.randn(*output.shape)
    grad_input = conv.backward(grad_output, x)
    print("Conv2d gradient shape:", grad_input.shape)

    # Test Linear
    linear = Linear(in_features=784, out_features=10)
    x = np.random.randn(32, 784)  # batch_size=32, features=784

    # Forward pass
    output = linear(x)
    print("Linear output shape:", output.shape)

    # Backward pass
    grad_output = np.random.randn(*output.shape)
    grad_input = linear.backward(grad_output, x)
    print("Linear gradient shape:", grad_input.shape)


    # conv = Conv2d(3, 64, (4,4), 2, 'same', padding_mode='reflect')
    # norm = BatchNorm2d(64)
    # relu = ReLU()
    # pool = MaxPool2d()
    # flatten = Flatten()
    # fc = Linear(64*4*4, 10)
    # softmax = Softmax()
    #
    # x = np.random.randn(4, 3, 16, 16)
    # print("Input shape:", x.shape)
    # x = conv(x)
    # print("Conv shape:", x.shape)
    # x = norm(x)
    # print("Norm shape:", x.shape)
    # x = relu(x)
    # print("ReLU shape:", x.shape)
    # x = pool(x)
    # print("Pool shape:", x.shape)
    # x = flatten(x)
    # print("Flatten shape:", x.shape)
    # x = fc(x)
    # print("FC shape:", x.shape)
    # x = softmax(x)
    # print("Softmax shape:", x.shape)