from torchclone.optim import Adam
from torchclone.losses import CrossEntropyLoss
import torchclone.modules as md
import numpy as np
from torchclone.model import Model
from torchclone.data import Dataset, DataLoader

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
    input_shape = (32, 3, 32, 32)
    num_classes = 10

    dummy_inputs = np.random.randn(*input_shape)
    dummy_labels = np.random.randint(0, num_classes, input_shape[0])

    validation_inputs = np.random.randn(*input_shape)
    validation_labels = np.random.randint(0, num_classes, input_shape[0])

    # Create datasets and dataloaders
    batch_size = 8
    dataset = Dataset(dummy_inputs, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    val_dataset = Dataset(validation_inputs, validation_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Train model
    model.train(optimizer, criterion, dataloader, num_epochs=20, val=val_dataloader)

    # Evaluate model
    model.eval(dataloader, criterion)

    # Predict
    y_hat = model.predict(dummy_inputs)
    pred_classes = np.argmax(y_hat, axis=1)

    print("Predicted: " ,pred_classes)
    print("Actual:    ", dummy_labels)