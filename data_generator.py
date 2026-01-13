"""Custom dataset generation options."""

import numpy as np


class DataGenerator:
    """Generate different types of synthetic datasets."""
    
    @staticmethod
    def linear(n_points=50, slope=1.5, intercept=2.0, noise_std=1.5, seed=None):
        """Generate linear dataset with optional noise."""
        if seed is not None:
            np.random.seed(seed)
        X = np.linspace(0, 10, n_points)
        Y = slope * X + intercept + np.random.normal(0, noise_std, n_points)
        return X, Y

    @staticmethod
    def polynomial(n_points=50, coefficients=[1, 0.5, 0.1], noise_std=1.0, seed=None):
        """Generate polynomial dataset."""
        if seed is not None:
            np.random.seed(seed)
        X = np.linspace(0, 10, n_points)
        Y = np.polyval(coefficients, X) + np.random.normal(0, noise_std, n_points)
        return X, Y

    @staticmethod
    def sinusoidal(n_points=50, amplitude=2, frequency=0.5, noise_std=0.5, seed=None):
        """Generate sinusoidal dataset."""
        if seed is not None:
            np.random.seed(seed)
        X = np.linspace(0, 10, n_points)
        Y = amplitude * np.sin(frequency * X) + 5 + np.random.normal(0, noise_std, n_points)
        return X, Y

    @staticmethod
    def exponential(n_points=50, base=0.9, noise_std=1.0, seed=None):
        """Generate exponential dataset."""
        if seed is not None:
            np.random.seed(seed)
        X = np.linspace(0, 10, n_points)
        Y = 10 * np.exp(-base * X) + np.random.normal(0, noise_std, n_points)
        return X, Y

    @staticmethod
    def noisy_linear(n_points=50, noise_level=0.5, seed=None):
        """Generate linear data with configurable noise level."""
        if seed is not None:
            np.random.seed(seed)
        X = np.linspace(0, 10, n_points)
        Y = 1.5 * X + 2.0 + np.random.normal(0, noise_level * 2, n_points)
        return X, Y
