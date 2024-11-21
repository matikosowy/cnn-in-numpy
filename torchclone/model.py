from tqdm import tqdm
from typing import List, Tuple, Union
from torchclone.optim import SGD, Adam
from torchclone.losses import CrossEntropyLoss, MSELoss, L1Loss, BinaryCrossEntropyLoss
import numpy as np
from torchclone.data import DataLoader


class Model:
    """
    Model class to hold layers and parameters.

    Parameters:
    -----------
    layers: List[Layer]
        List of layers in the model
    """
    def __init__(self, layers: List):
        self.layers = layers
        self.parameters = []
        self.training = True

        # Collect trainable parameters
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                self.parameters.extend([layer.weight, layer.bias])

    def __str__(self):
        return "Model's layers:\n--> " + '\n--> '.join([str(layer) for layer in self.layers])
    
    def _train_mode(self):
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
                
    def _eval_mode(self):
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def train(self, optimizer: Union[SGD, Adam],criterion: Union[MSELoss, L1Loss, CrossEntropyLoss,
        BinaryCrossEntropyLoss], dataloader: DataLoader, num_epochs: int, val: DataLoader=None):
        """
        Train the model.

        Parameters:
        -----------
        optimizer: Union[SGD, Adam]
            Optimizer to use for training
        criterion: Union[CrossEntropyLoss, MSELoss, L1Loss, BinaryCrossEntropyLoss]
            Loss function to use
        dataloader: DataLoader
            DataLoader object to iterate over the dataset
        num_epochs: int
            Number of epochs to train
        """
        self._train_mode()
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
                for batch in dataloader:
                    avg_loss = train_one_epoch(self, optimizer, criterion, [batch])
                    epoch_loss += avg_loss
                    pbar.update(1)

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_epoch_loss:.4f}")

            if val is not None:
                self._eval_mode()
                val_loss = self.eval(val, criterion)
                self._train_mode()
                print(f"Val Loss: {val_loss:.4f}")


    def eval(self, dataloader: DataLoader, criterion: Union[MSELoss, L1Loss, CrossEntropyLoss,
            BinaryCrossEntropyLoss]):
            """
            Evaluate the model.

            Parameters:
            -----------
            dataloader: DataLoader
                DataLoader object to iterate over the dataset
            criterion: Union[CrossEntropyLoss, MSELoss, L1Loss, BinaryCrossEntropyLoss]
                Loss function to use
            """
            total_loss = 0.0

            for batch in dataloader:
                x, y = batch

                for layer in self.layers:
                    x = layer(x)

                loss = criterion.forward(x, y)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            return avg_loss

    def predict(self, x: np.ndarray):
        """
        Predict the output of the model.

        Parameters:
        -----------
        x: np.ndarray
            Input data
        """
        for layer in self.layers:
            x = layer(x)

        return x


def train_one_epoch(model: Model,
                optimizer: Union[SGD, Adam],
                criterion: Union[CrossEntropyLoss, MSELoss, L1Loss, BinaryCrossEntropyLoss],
                dataloader: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Train the model for one epoch.

    Parameters:
    -----------
    model: Model
        Model to train

    optimizer: Union[SGD, Adam]
        Optimizer to use for training

    criterion: Union[CrossEntropyLoss, MSELoss]
        Loss function to use

    dataloader: List[Tuple[np.ndarray, np.ndarray]]
        List of input-target pairs
    """
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        x = inputs
        layer_outputs = []

        for layer in model.layers:
            x = layer(x)
            layer_outputs.append(x)

        # Compute loss
        loss = criterion.forward(x, targets)
        total_loss += loss

        # Backward pass
        grad = criterion.backward()

        for layer, output in zip(reversed(model.layers), reversed(layer_outputs[:-1])):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)

        # Update parameters
        gradients = []
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                gradients.extend([layer.weight_grad, layer.bias_grad])

        optimizer.step(gradients)

    return total_loss / len(dataloader)