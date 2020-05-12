from typing import Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error


class ILC:
    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

        self.t = None
        self.y = None

    def update(self, t: np.ndarray, x: np.ndarray, y: np.ndarray,
               xd: np.ndarray = None, yd: np.ndarray = None):
        if self.t is None:
            self.t = t

        x_interp = np.interp(self.t, t, x)
        y_interp = np.interp(self.t, t, y)
        error = y_interp - x_interp
        error_d = np.zeros(error.shape)

        if xd is not None and yd is not None:
            xd_interp = np.interp(self.t, t, xd)
            yd_interp = np.interp(self.t, t, yd)
            error_d = yd_interp - xd_interp

        if self.y is None:
            self.y = np.zeros(y.shape)

        self.y += self.kp * error + self.kd * error_d

    def control(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.t, self.y)

    def mean_error(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, delay: float = 0.0):
        x_interp = np.interp(self.t, t - delay , x)
        y_interp = np.interp(self.t, t, y)

        return mean_absolute_error(y_interp, x_interp)

    def estimate_delay(self, t: np.ndarray, x: np.ndarray, y: np.ndarray):
        limit = 0.5 * (t.min() + t.max())
        delays = np.arange(-limit, limit, 0.01)

        errors = [self.mean_error(t, x, y, delay) for delay in delays]
        index = np.argmin(errors)

        return delays[index]
