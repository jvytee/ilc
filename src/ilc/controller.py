import numpy as np


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
        e = y_interp - x_interp
        ed = np.zeros(e.shape)

        if xd is not None and yd is not None:
            xd_interp = np.interp(self.t, t, xd)
            yd_interp = np.interp(self.t, t, yd)
            ed = yd_interp - xd_interp

        if self.y is None:
            self.y = np.zeros(y.shape)

        self.y += self.kp * e + self.kd * ed

    def control(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.t, self.y)
