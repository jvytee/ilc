import numpy as np
from sklearn.metrics import mean_absolute_error


class ILC:
    """Simple implementation of an Incremental Learning Controller

    Parameters
    ----------
    kp : float
        Proportional control gain
    kd : float
        Derivative control gain
    """

    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

        self.t = None
        self.y = None

    def update(self, t: np.ndarray, x: np.ndarray, y: np.ndarray,
               xd: np.ndarray = None, yd: np.ndarray = None):
        """Updates the controller using the current result and reference signal

        Parameters
        ----------
        t : np.ndarray
            Sample times of current/reference data
        x : np.ndarray
            Samples representing the current result
        y : np.ndarray
            Samples representing the reference signal
        xd : np.ndarray
            Optional samples representing the first derivative of the current result
        yd : np.ndarray
            Optional samples representing the first derivative of the reference signal
        """
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
        """Returns the learnt control signal at given times

        Parameters
        ----------
        t : np.ndarray
            Array of times at which to sample the control signal

        Returns
        -------
        np.ndarray
            Control signal values sampled at given times
        """
        return np.interp(t, self.t, self.y)

    def mean_error(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, delay: float = 0.0) -> float:
        """Estimates the mean error for given result and reference samples

        Parameters
        ----------
        t : np.ndarray
            Sample times of current/reference data
        x : np.ndarray
            Samples representing the current result
        y : np.ndarray
            Samples representing the reference signal
        delay : float
            Optional delay to subtract from result samples

        Returns
        -------
        float
            Mean absolute error
        """
        x_interp = np.interp(self.t, t - delay, x)
        y_interp = np.interp(self.t, t, y)

        return mean_absolute_error(y_interp, x_interp)

    def estimate_delay(self, t: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Estimates the delay of result samples

        Parameters
        ----------
        t : np.ndarray
            Sample times of current/reference data
        x : np.ndarray
            Samples representing the current result
        y : np.ndarray
            Samples representing the reference signal
        """
        limit = 0.5 * (t.min() + t.max())
        delays = np.arange(-limit, limit, 0.01)

        errors = [self.mean_error(t, x, y, delay) for delay in delays]
        index = np.argmin(errors)

        return delays[index]
