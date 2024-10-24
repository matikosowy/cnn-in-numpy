from optim import Adam, CrossEntropyLoss
import modules as md
import numpy as np
from model import Model

if __name__ == '__main__':
    # Define model
    model = Model([
        md.Conv2d(3, 64, 3, stride=1, padding='same'),
        md.BatchNorm2d(64),
        md.ReLU(),
        md.MaxPool2d(2),
        md.Flatten(),
        md.Linear(64 * 16 * 16, 10)
    ])

    print(model)

    optimizer = Adam(model.parameters, lr=1e-3)

    # Create loss function
    criterion = CrossEntropyLoss()

    # Create dummy data
    batch_size = 32
    input_shape = (batch_size, 3, 32, 32)
    num_classes = 10

    dummy_data = [
        (np.random.randn(*input_shape),
         np.random.randint(0, num_classes, size=batch_size))
        for _ in range(10)
    ]

    # Train model
    model.train(optimizer, criterion, dummy_data, num_epochs=5)