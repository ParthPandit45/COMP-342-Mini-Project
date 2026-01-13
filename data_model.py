import numpy as np
import config as cfg


class DataModel:
    def __init__(self, seed=42):
        self.seed = seed
        self.X = None
        self.Y = None
        self.N = 0
        self.X_MAX = self.X_MIN = 0.0
        self.Y_MAX = self.Y_MIN = 0.0
        self.Y_RANGE = 1.0
        self.generate(seed=self.seed)

    def generate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        self.X = np.linspace(0, 10, cfg.N_POINTS)
        self.Y = cfg.TRUE_S * self.X + cfg.TRUE_I + np.random.normal(0, cfg.NOISE_STD, cfg.N_POINTS)
        self.N = len(self.X)
        self.X_MAX, self.X_MIN = self.X.max(), self.X.min()
        self.Y_MAX, self.Y_MIN = self.Y.max(), self.Y.min()
        self.Y_RANGE = max(self.Y_MAX - self.Y_MIN, 1e-6)

    def data_to_canvas(self, x_data, y_data):
        plot_w, plot_h = cfg.CANVAS_W - 2 * cfg.PAD, cfg.CANVAS_H - 2 * cfg.PAD
        x_pixel = cfg.PAD + (x_data - self.X_MIN) * (plot_w / (self.X_MAX - self.X_MIN))
        y_norm = (y_data - self.Y_MIN) / self.Y_RANGE
        y_pixel = cfg.CANVAS_H - cfg.PAD - y_norm * plot_h
        return x_pixel, y_pixel
