import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional
from scipy.sparse.linalg import svds



class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()
    MSE_regularized = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE,
        regularization: float = 0.0,
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []
        self.mu = regularization
        self.grad = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            self.grad = 2 / X.shape[0] * X.T @ (X @ self.w - y)
            return self.grad
        elif self.loss_function is LossFunction.MSE_regularized:
            self.grad = 2 / X.shape[0] * X.T @ (X @ self.w - y) + self.mu * self.w
            return self.grad
        return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function is LossFunction.MSE:
            res = X @ self.w - y
            return float(np.dot(res, res) / X.shape[0])
        elif self.loss_function is LossFunction.MSE_regularized:
            res = X @ self.w - y
            return float(np.dot(res, res) / X.shape[0]) + self.mu * 0.5 * np.dot(self.w, self.w)
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.y_train = X, y
        self.w = np.zeros(X.shape[1])

        self.loss_history = [self.compute_loss(X, y)]

        if isinstance(self.optimizer, BaseDescent):
            for _ in range(self.max_iter):
                # 1 шаг градиентного спуска

                delta = self.optimizer.step()
                self.loss_history.append(self.compute_loss(X, y))

                if np.isnan(delta).any():
                    break

                elif np.linalg.norm(delta) < self.tolerance:
                    break

                elif self.grad is not None and np.linalg.norm(self.grad) < self.tolerance:
                    break
    


        elif self.optimizer is None:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.optimizer == 'SVD':
            U, S, vt = svds(X, k=4)
            self.w = vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y


        return self
