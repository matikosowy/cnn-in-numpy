import numpy as np
from typing import Tuple, Optional

class CrossEntropyLoss:
    """Cross Entropy loss"""
    def __init__(self):
        self.cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes cross entropy loss.

        Parameters:
        -----------
        pred: np.ndarray
            Predicted values

        target: np.ndarray
            Target values

        Returns:
        --------
        float
            Average loss over the batch
        """
        batch_size = pred.shape[0]
        # Add small epsilon to avoid log(0)
        pred = np.clip(pred, 1e-7, 1 - 1e-7)

        # Convert target to one-hot encoding
        target_one_hot = np.zeros_like(pred)
        target_one_hot[np.arange(batch_size), target] = 1

        # Compute cross entropy loss
        loss = -np.sum(target_one_hot * np.log(pred)) / batch_size

        # Save in cache for backward pass
        self.cache = (pred, target_one_hot)

        return loss

    def backward(self) -> np.ndarray:
        """
        Computes gradient of loss with respect to predictions.

        Returns:
        --------
        np.ndarray
            Gradient of same shape as predictions
        """
        pred, target_one_hot = self.cache
        return (pred - target_one_hot) / pred.shape[0]
    
    
class BinaryCrossEntropyLoss:
    """Binary Cross Entropy loss"""
    def __init__(self):
        self.cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes binary cross entropy loss.

        Parameters:
        -----------
        pred: np.ndarray
            Predicted values, expected to be in range [0, 1]

        target: np.ndarray
            Target values, expected to be either 0 or 1

        Returns:
        --------
        float
            Average loss over the batch
        """
        batch_size = pred.shape[0]
        # Add small epsilon to avoid log(0)
        pred = np.clip(pred, 1e-7, 1 - 1e-7)

        # Compute cross entropy loss
        loss = -np.sum(target * np.log(pred) + (1 - target) * np.log(1 - pred)) / batch_size

        # Save in cache for backward pass
        self.cache = (pred, target)

        return loss

    def backward(self) -> np.ndarray:
        """
        Computes gradient of loss with respect to predictions.

        Returns:
        --------
        np.ndarray
            Gradient of same shape as predictions
        """
        pred, target = self.cache
        return (pred - target) / (pred / (1 - pred)) / pred.shape[0]


class MSELoss:
    """Mean Squared Error loss"""
    def __init__(self):
        self.cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes mean squared error loss.

        Parameters:
        -----------
        pred: np.ndarray
            Predicted values

        target: np.ndarray
            Target values

        Returns:
        --------
        float
            Average loss over the batch
        """
        self.cache = (pred, target)
        return np.mean((pred - target) ** 2)

    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss with respect to predictions.

        Returns:
            Gradient of same shape as predictions
        """
        pred, target = self.cache
        return 2 * (pred - target) / pred.size


class L1Loss:
    """ L1 Norm / Mean Absolute Error loss"""
    def __init__(self):
        self.cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes L1 loss.

        Parameters:
        -----------
        pred: np.ndarray
            Predicted values

        target: np.ndarray
            Target values

        Returns:
        --------
        float
            Average loss over the batch
        """
        self.cache = (pred, target)
        return np.mean(np.abs(pred - target))

    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss with respect to predictions.

        Returns:
            Gradient of same shape as predictions
        """
        pred, target = self.cache
        return np.sign(pred - target) / pred.size