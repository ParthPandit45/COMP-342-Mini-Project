"""Metrics and statistics tracking for the regression model."""

import numpy as np
import sklearn.metrics as metrics


class MetricsTracker:
    def __init__(self):
        self.mse_history = []
        self.mae_history = []
        self.r2_history = []
        self.iteration_history = []
        self.current_iteration = 0

    def update(self, y_true, y_pred, iteration):
        self.current_iteration = iteration
        self.iteration_history.append(iteration)
        self.mse_history.append(metrics.mean_squared_error(y_true, y_pred))
        self.mae_history.append(metrics.mean_absolute_error(y_true, y_pred))

        # Calculate R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        self.r2_history.append(r2)

    def get_current(self):
        if not self.mse_history:
            return 0, 0, 0
        return self.mse_history[-1], self.mae_history[-1], self.r2_history[-1]

    def clear(self):
        self.mse_history.clear()
        self.mae_history.clear()
        self.r2_history.clear()
        self.iteration_history.clear()
        self.current_iteration = 0

    def summary(self):
        """Return a summary of training statistics."""
        if not self.mse_history:
            return {}
        return {
            "iterations": len(self.mse_history),
            "final_mse": self.mse_history[-1],
            "min_mse": min(self.mse_history),
            "max_mse": max(self.mse_history),
            "final_mae": self.mae_history[-1],
            "final_r2": self.r2_history[-1],
        }
