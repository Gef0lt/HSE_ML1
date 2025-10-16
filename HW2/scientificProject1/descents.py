import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.iteration += 1
        w = self.update_weights()
        return w


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        self.model.w -= self.lr_schedule.get_lr(self.iteration) * gradient

        return self.lr_schedule.get_lr(self.iteration) * gradient


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        batch = np.random.choice(self.model.X_train.shape[0], size=self.batch_size, replace=False)
        if hasattr(X_train, 'iloc'):
            X_batch = X_train.iloc[batch].to_numpy() if hasattr(X_train.iloc[batch], 'to_numpy') else X_train.iloc[batch].values
        else:
            X_batch = X_train[batch]

        if hasattr(y_train, 'iloc'):
            y_batch = y_train.iloc[batch].to_numpy() if hasattr(y_train.iloc[batch], 'to_numpy') else y_train.iloc[batch].values
        else:
            y_batch = y_train[batch]

        grad = self.model.compute_gradients(X_batch, y_batch)
        self.model.w -= self.lr_schedule.get_lr(self.iteration) * grad
        return self.lr_schedule.get_lr(self.iteration) * grad



class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None
        self.g_bar = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train

        if self.grad_memory is None:
            self.grad_memory = np.zeros(X_train.shape, dtype=float)
            self.g_bar = 0

        j = np.random.choice(X_train.shape[0])
        g_new = self.model.compute_gradients(X_train[j:j+1], y_train[j:j+1])
        g_old = self.grad_memory[j].copy()
        self.grad_memory[j] = g_new
        self.g_bar += (g_new - g_old) / X_train.shape[0]
        self.model.w -= self.lr_schedule.get_lr(self.iteration) * self.g_bar

        return self.lr_schedule.get_lr(self.iteration) * self.g_bar

class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR,
                 beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients(self.model.X_train,
                                                self.model.y_train)
        self.velocity = self.velocity * self.beta + self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w -= self.velocity

        return self.velocity

class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        if self.v is None:
            self.v = np.zeros_like(self.model.w)
        if self.m is None:
            self.m = np.zeros_like(self.model.w)

        grad = self.model.compute_gradients(self.model.X_train, self.model.y_train)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        v_cap = self.v / (1 - self.beta2 ** (self.iteration))
        m_cap = self.m / (1 - self.beta1 ** (self.iteration))
        self.model.w -= self.lr_schedule.get_lr(self.iteration) * m_cap / (np.sqrt(v_cap) + self.eps)

        return self.lr_schedule.get_lr(self.iteration) * m_cap / (np.sqrt(v_cap) + self.eps)
