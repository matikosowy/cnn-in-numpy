from typing import List, Tuple
import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Parameters:
    -----------
    parameters: List[np.ndarray]
        List of parameters to optimize
    lr: float
        Learning rate
    momentum: float
        Momentum factor
    """
    def __init__(self,
                 parameters: List[np.ndarray],
                 lr: float = 0.01,
                 momentum: float = 0.0
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in parameters]

    def step(self, gradients: List[np.ndarray]):
        """
        Update parameters using SGD with momentum

        Parameters:
        -----------
        gradients: List[np.ndarray]
            List of gradients for each parameter
        """
        for p, v, g in zip(self.parameters, self.velocities, gradients):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    """
    Adam optimizer.

    Parameters:
    -----------
    parameters: List[np.ndarray]
        List of parameters to optimize
    lr: float
        Learning rate
    betas: Tuple[float, float]
        Decay rates for first and second moment
    eps: float
        Small value to avoid division by zero
    """
    def __init__(self,
                 parameters: List[np.ndarray],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.m = [np.zeros_like(p) for p in parameters]  # First moment
        self.v = [np.zeros_like(p) for p in parameters]  # Second moment
        self.t = 0  # Time step

    def step(self, gradients: List[np.ndarray]):
        """
        Update parameters using Adam optimizer

        Parameters:
        -----------
        gradients: List[np.ndarray]
            List of gradients for each parameter
        """
        self.t += 1

        for i, (p, g) in enumerate(zip(self.parameters, gradients)):
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
