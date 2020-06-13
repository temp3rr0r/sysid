"""
This module performs system identification.
"""

import cupy as np
import scipy
import numpy

import matplotlib.pyplot as plt

__all__ = ['StateSpaceDiscreteLinear',
           'StateSpaceDataList', 'StateSpaceDataArray']


class StateSpaceDiscreteLinear(object):
    """
    State space for discrete linear systems.
    """

    def __init__(self, A, B, C, D, Q, R, dt):

        self.dt = dt

        self.A = np.array(A, dtype=numpy.float16)
        self.B = np.array(B, dtype=numpy.float16)
        self.C = np.array(C, dtype=numpy.float16)
        self.D = np.array(D, dtype=numpy.float16)
        self.Q = np.array(Q, dtype=numpy.float16)
        self.R = np.array(R, dtype=numpy.float16)
        n_x = A.shape[0]
        n_u = B.shape[1]
        n_y = C.shape[0]

        assert self.A.shape[1] == n_x
        assert self.B.shape[0] == n_x
        assert self.C.shape[1] == n_x
        assert self.D.shape[0] == n_y
        assert self.D.shape[1] == n_u
        assert self.Q.shape[0] == n_x
        assert self.Q.shape[1] == n_x
        # assert self.R.shape[0] == n_u
        # assert self.R.shape[1] == n_u
        assert self.R.shape[0] == n_y  # TODO
        assert self.R.shape[1] == n_y  # TODO
        # assert np.matrix(dt).shape == (1, 1)

    def dynamics(self, x, u, w):
        """
        Dynamics
        x(k+1) = A x(k) + B u(k) + w(k)
        E(ww^T) = Q

        Args:
            x (): The current state.
            u (): The current input.
            w (): The current process noise.

        Returns:
            x(k+1): The next state.

        """
        return self.A @ x + self.B @ u + w

    def measurement(self, x, u, v):
        """
        Measurement.
        y(k) = C x(k) + D u(k) + v(k)
        E(vv^T) = R

        Args:
            x (): The current state.
            u (): The current input.
            w (): The current process noise.

        Returns:
            y(k): The current measurement

        """

        # assert x.shape[1] == 1  # TODO
        # assert u.shape[1] == 1  # TODO
        # assert v.shape[1] == 1  # TODO
        return self.C @ x + self.D @ u + v

    def simulate(self, f_u, x0, tf):
        """
        Simulate the system.

        Args:
            f_u (): The input function  f_u(t, x, i)
            x0 (): The initial state.
            tf (): The final time.

        Returns:
            data: A StateSpaceDataArray object.

        """

        # assert x0.shape[1] == 1  # TODO
        t = 0  # TODO: make a full sized ndarray, no lists
        x = x0
        dt = self.dt
        data = StateSpaceDataList([], [], [], [])
        i = 0

        n_x = self.A.shape[0]
        n_u = self.B.shape[1]
        n_y = self.C.shape[0]

        # assert np.matrix(f_u(0, x0, 0)).shape[1] == 1  # TODO
        # assert np.matrix(f_u(0, x0, 0)).shape[0] == n_u  # TODO

        # take square root of noise cov to prepare for noise sim
        if np.linalg.norm(self.Q) > 0:
            sqrtQ = np.array(scipy.linalg.sqrtm(self.Q.get()), dtype=numpy.float16)
        else:
            sqrtQ = self.Q

        if np.linalg.norm(self.R) > 0:
            sqrtR = np.array(scipy.linalg.sqrtm(self.R.get()), dtype=numpy.float16)
        else:
            sqrtR = self.R

        # main simulation loop
        while t + dt < tf:
            u = f_u(t, x, i)
            v = sqrtR.dot(np.random.randn(n_y, 1))
            y = self.measurement(x, u, v)
            data.append(t, x, y, u)
            w = sqrtQ.dot(np.random.randn(n_x, 1))
            x = self.dynamics(x, u, w)
            t += dt
            i += 1
        return data.to_StateSpaceDataArray()

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


class StateSpaceDataList(object):
    """
    An expandable state space data list.
    """

    def __init__(self, t, x, y, u):

        self.t = t
        self.x = x
        self.y = y
        self.u = u

    def append(self, t, x, y, u):
        """
        Append time-step, state, output and input to lists.
        Args:
            t (): Time-tep.
            x (): State.
            y (): Output.
            u (): Input

        Returns: Nothing.

        """
        self.t += [t]
        self.x += [x]
        self.y += [y]
        self.u += [u]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def to_StateSpaceDataArray(self):
        """
        Converts to a state space data array object.
        With fixed sizes.

        Returns: StateSpaceDataArray with nd arrays instead of lists.

        """
        ssd1 = StateSpaceDataArray(
            t=np.array(self.t, dtype=numpy.float16).T,
            x=np.array(self.x, dtype=numpy.float16).T,
            y=np.array(self.y, dtype=numpy.float16).T,
            u=np.array(self.u, dtype=numpy.float16).T)
        return ssd1


class StateSpaceDataArray(object):
    """
    A fixed size state space data list.
    """

    def __init__(self, t, x, y, u):

        self.t = t
        self.x = x
        self.y = y
        self.u = u

        # assert self.t.shape[0] == 1
        # assert self.x.shape[0] < self.x.shape[1]
        # assert self.y.shape[0] < self.y.shape[1]
        # assert self.u.shape[0] < self.u.shape[1]

    def to_StateSpaceDataList(self):
        """
        Convert to StateSpaceDataList that you can append to.
        Returns: List of nd arrays.

        """
        return StateSpaceDataList(
            t=list(self.t),
            x=list(self.x),
            y=list(self.y),
            u=list(self.u))

    def plot(self, plot_x=False, plot_y=False, plot_u=False):
        """
        Plot data.
        Args:
            plot_x (bool): Set True to plot states.
            plot_y (bool): Set True to plot outputs.
            plot_u (bool): Set True to plot inputs.

        Returns: Nothing.

        """
        t = self.t.T
        x = self.x.T
        y = self.y.T
        u = self.u.T
        if plot_x:
            plt.plot(t, x)
        if plot_y:
            plt.plot(t, y)
        if plot_u:
            plt.plot(t, u)
