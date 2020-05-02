import numpy as np


class IncrementalLearner:
    def __init__(self):
        self.x_update = list()
        self.y_update = list()
        self.yd_update = list()
        self.x = None
        self.y = None
        self.yd = None

    def apply_update(self):
        if self.x is None:
            self.x = self.x_update

        if self.y is None:
            self.y = self.y_update
        else:
            self.y = np.interp(self.x, self.x_update, self.y_update)

        if self.yd is None:
            self.yd = self.yd_update
        else:
            self.yd = np.interp(self.x, self.x_update, self.yd_update)

    def get(self, x: np.ndarray) -> np.ndarray:
        return np.stack([
            np.interp(x, self.x, self.y),
            np.interp(x, self.x, self.yd)
        ]).T
