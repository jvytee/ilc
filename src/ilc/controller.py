import numpy as np


class ILC:
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd
        
        self.x = None
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray, yd: np.ndarray = None):
        if self.x is None:
            self.x = x

        if self.y is None:
            self.y = y
        else:
            self.y += self.kp * np.interp(self.x, x, y) + self.kd * np.interp(self.x, x, yd)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.stack([
            np.interp(x, self.x, self.y),
            np.interp(x, self.x, self.yd)
        ]).T
