"""Optimization algorithms for training."""

import numpy as np


class Optimizer:
    """Base optimizer class."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, slope, intercept, d_slope, d_intercept):
        """Perform one optimization step."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def step(self, slope, intercept, d_slope, d_intercept):
        new_slope = slope - self.learning_rate * d_slope
        new_intercept = intercept - self.learning_rate * d_intercept
        return new_slope, new_intercept


class Momentum(Optimizer):
    """SGD with momentum."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_slope = 0
        self.velocity_intercept = 0

    def step(self, slope, intercept, d_slope, d_intercept):
        self.velocity_slope = self.momentum * self.velocity_slope - self.learning_rate * d_slope
        self.velocity_intercept = self.momentum * self.velocity_intercept - self.learning_rate * d_intercept

        new_slope = slope + self.velocity_slope
        new_intercept = intercept + self.velocity_intercept
        return new_slope, new_intercept

    def reset(self):
        self.velocity_slope = 0
        self.velocity_intercept = 0


class Adam(Optimizer):
    """Adaptive Moment Estimation optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_slope = 0
        self.m_intercept = 0
        self.v_slope = 0
        self.v_intercept = 0
        self.t = 0

    def step(self, slope, intercept, d_slope, d_intercept):
        self.t += 1

        # Update biased first moment estimate
        self.m_slope = self.beta1 * self.m_slope + (1 - self.beta1) * d_slope
        self.m_intercept = self.beta1 * self.m_intercept + (1 - self.beta1) * d_intercept

        # Update biased second raw moment estimate
        self.v_slope = self.beta2 * self.v_slope + (1 - self.beta2) * (d_slope ** 2)
        self.v_intercept = self.beta2 * self.v_intercept + (1 - self.beta2) * (d_intercept ** 2)

        # Compute bias-corrected first moment estimate
        m_hat_slope = self.m_slope / (1 - self.beta1 ** self.t)
        m_hat_intercept = self.m_intercept / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat_slope = self.v_slope / (1 - self.beta2 ** self.t)
        v_hat_intercept = self.v_intercept / (1 - self.beta2 ** self.t)

        # Update parameters
        new_slope = slope - self.learning_rate * m_hat_slope / (np.sqrt(v_hat_slope) + self.epsilon)
        new_intercept = intercept - self.learning_rate * m_hat_intercept / (
            np.sqrt(v_hat_intercept) + self.epsilon
        )

        return new_slope, new_intercept

    def reset(self):
        self.m_slope = 0
        self.m_intercept = 0
        self.v_slope = 0
        self.v_intercept = 0
        self.t = 0
