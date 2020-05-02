from typing import Union, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error
from dmp.Dmp import Dmp
from dmp.Trajectory import Trajectory
from functionapproximators.FunctionApproximatorLWR import FunctionApproximatorLWR


class IncrementalLearner:
    """A simple implementation of Incremental Learning Control (ILC)"""

    def __init__(self, task_parameters: np.ndarray):
        """Creates a new IncrementalLearner instance

        Parameters
        ----------
        task_parameters : np.ndarray
            Parameters of the task to be lernt
        """
        self.task_parameters = task_parameters
        self.times = list()
        self.q_sampled = list()
        self.q_desired = list()
        self.dq_sampled = list()
        self.dq_desired = list()
        self.dmp = None
        self.values = None
        self.mean_error = None

    def collect(self, time: float, q_sampled: np.ndarray, q_desired: np.ndarray, dq_sampled: np.ndarray,
                dq_desired: np.array) -> None:
        """Adds a new training sample for the controller to learn on

        Parameters
        ----------
        time : float
            Capture time of the sample
        q_sampled : np.ndarray
            Sampled joint space configuration
        q_desired : np.ndarray
            Desired joint space configuration
        dq_sampled : np.ndarray
            Sampled joint space velocity
        dq_desired : np.ndarray
            Desired joint space velocity
        """
        if len(q_sampled) == len(q_desired) and len(dq_sampled) == len(dq_desired):
            self.times.append(time)
            self.q_sampled.append(q_sampled)
            self.q_desired.append(q_desired)
            self.dq_sampled.append(dq_sampled)
            self.dq_desired.append(dq_desired)

    def train(self, kp: float = 12, kd: float = 4) -> bool:
        """Trains an internal DMP on the newest samples

        Parameters
        ----------
        kp : float
            Proportional gain for training
        kd : float
            Derivative gain for training

        Returns
        -------
        bool
            True if DMP was successfully trained on collected data
        """
        if not self.times:
            return False

        times = np.array(self.times)
        q_sampled = np.array(self.q_sampled)
        q_desired = np.array(self.q_desired)
        dq_sampled = np.array(self.dq_sampled)
        dq_desired = np.array(self.dq_desired)

        self.times = list()
        self.q_sampled = list()
        self.q_desired = list()
        self.dq_sampled = list()
        self.dq_desired = list()

        delay = self.find_delay(q_desired, q_sampled, (0, q_sampled.shape[0] // 2))
        self.mean_error = self.estimate_mean_error(q_desired, q_sampled, 0)

        previous = np.zeros(q_sampled.shape)
        if self.values is not None:
            previous = np.array([self.query(time) for time in times])

        error = q_desired - q_sampled
        error_derivative = dq_desired - dq_sampled

        if delay >= 0:
            error_delayed = np.array([*error[delay:], *(np.ones((delay, error.shape[1])) * error[-1])])
            error_derivative_delayed = np.array(
                [*error_derivative[delay:], *(np.ones((delay, error_derivative.shape[1])) * error_derivative[-1])])
        else:
            error_delayed = np.array([*(np.ones((-delay, error.shape[1])) * error[0]), *error[:delay]])
            error_derivative_delayed = np.array(
                [*(np.ones((-delay, error_derivative.shape[1])) * error_derivative[0]), *error_derivative[:delay]])

        signals = previous + kp * error_delayed + kd * error_derivative_delayed
        trajectory = Trajectory(times, signals)
        approximators = [FunctionApproximatorLWR(10) for _ in signals[0]]
        self.dmp = Dmp(tau=times[-1],
                       y_init=signals[0],
                       y_attr=signals[-1],
                       function_apps=approximators,
                       sigmoid_max_rate=-1,
                       forcing_term_scaling="G_MINUS_Y0_SCALING")
        self.dmp.train(trajectory)

        return True

    def integrate(self, tau: float, dt: float) -> bool:
        """Integrates the dynamical system of the trained DMP and updates internal values

        Parameters
        ----------
        tau : float
            Total time to integrate over
        dt : float
            Timestep for integration

        Returns
        -------
        bool
            True if successful, False otherwise
        """

        if self.dmp is None:
            return False

        ts = np.linspace(0, tau, int(tau / dt) + 1)
        states = self.dmp.analyticalSolution(ts)
        trajectory = self.dmp.statesAsTrajectory(ts, states[0], states[1])
        self.values = trajectory.asMatrix()

        return True

    def query(self, time: float) -> Union[np.ndarray, None]:
        """Get interpolated DMP value at given time.
        ILC needs to be trained & integrated.

        Parameters
        ----------
        time : float
            Time to query ILC at

        Returns
        -------
        Union[np.ndarray, None]
            Joint configuration at the given time.
            None if ILC has not been trained/integrated.
        """

        if self.values is None:
            return None

        ts = self.values[:, 0]
        xs = self.values[:, 1:self.dmp.dim_orig_ + 1].transpose()
        return np.array([np.interp(time, ts, dim) for dim in xs])

    def set_parameters(self, weights: np.ndarray, dmp: Dmp = None) -> None:
        """Initialize DMP with weights, e.g. from previous training

        Parameters
        ----------
        weights : np.ndarray
            Array of Gaussian weights (float)
        dmp : str
            A previously trained Dmp instance
        """

        # Load DMP if given
        if dmp is not None:
            self.dmp = dmp

        # Set weights if DMP available
        if self.dmp is not None:
            self.dmp.setParameterVectorSelected(weights)
            print('Set weights', self.dmp.getParameterVectorSelected())

    def estimate_mean_error(self, q_desired: np.ndarray, q_sampled: np.ndarray, delay: int = 0) -> float:
        """Compute mean deviation between q_desired and q_sampled with temporal delay using np.linalg.norm

        Parameters
        ----------
        q_desired : np.ndarray
            Array of desired values over time
        q_sampled : np.ndarray
            Array of actual values over time
        delay : int
            Delay as number of samples

        Returns
        -------
        float
            Estimated mean error
        """
        error = list()
        for i in range(-delay, q_sampled.shape[0]):
            desired = q_desired[0] if i < 0 else q_desired[i]
            sampled = q_sampled[-1] if i + delay >= q_sampled.shape[0] else q_sampled[i + delay]
            error.append(mean_absolute_error(desired, sampled))

        return np.mean(error)

    def find_delay(self, q_desired: np.ndarray, q_sampled: np.ndarray, interval: Tuple[int, int]) -> int:
        """Minimize mean error between q_desired and q_sampled to find delay between samples

        Parameters
        ----------
        q_desired : np.ndarray
            Array of desired values over time
        q_sampled : np.ndarray
            Array of actual values over time
        interval : Tuple[int, int]
            Tuple of two integers defining the search interval

        Returns
        -------
        int
            Estimated delay as number of samples
        """
        errors = [self.estimate_mean_error(q_desired, q_sampled, d) for d in range(*interval)]
        min_error_delay = np.argmin(errors)

        return np.arange(*interval)[min_error_delay]
