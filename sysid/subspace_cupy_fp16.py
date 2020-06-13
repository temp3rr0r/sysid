"""
This module performs subspace system identification.
It enforces that matrices are used instead of arrays to avoid dimension conflicts.
"""
import sysid
import cupy as np
import numpy
from sklearn.metrics import mean_squared_error, mean_absolute_error

__all__ = ['subspace_det_algo1', 'prbs', 'nrms', 'symmetric_mean_absolute_percentage_error',
           'mean_absolute_scaled_error', 'mean_absolute_percentage_error', 'index_of_agreement']


def block_hankel(data, f):
    """
    Create a block hankel matrix.
    Args:
        data (float): Array.
        f (int): number of rows

    Returns:
        Hankel matrix of f rows.

    """
    n = data.shape[1] - f
    return np.vstack([
        np.hstack([data[:, i + j] for i in range(f)])
        for j in range(n)]).T


def project(A):
    """
    Creates a projection matrix onto the row-space of A.
    Args:
        A (float): Matrix (not necessarily square).

    Returns:
        Projected matrix.

    """
    return A.T @ np.linalg.pinv(A @ A.T) @ A


def project_perp(A):
    """
    Creates a projection matrix onto the space perpendicular to the
    rowspace of A.

    Args:
        A ():

    Returns:

    """
    return np.eye(A.shape[1], dtype=numpy.float16) - project(A)


def project_oblique(B, C):
    """
    Projects along rowspace of B onto rowspace of C.
    """
    proj_B_perp = project_perp(B)
    return proj_B_perp @ np.linalg.pinv(C @ proj_B_perp) @ C


def subspace_det_algo1(y, u, f, p, s_tol, dt, order=-1):
    assert f > 1
    assert p > 1

    # setup matrices
    y = np.array(y, dtype=numpy.float16)
    n_y = y.shape[0]
    u = np.array(u, dtype=numpy.float16)
    n_u = u.shape[0]
    w = np.vstack([y, u])
    n_w = w.shape[0]

    # make sure the input is column vectors
    assert y.shape[0] < y.shape[1]
    assert u.shape[0] < u.shape[1]

    W = block_hankel(w, f + p)
    U = block_hankel(u, f + p)
    Y = block_hankel(y, f + p)

    W_p = W[:n_w*p, :]
    W_pp = W[:n_w*(p+1), :]

    Y_f = Y[n_y*f:, :]
    U_f = U[n_y*f:, :]

    Y_fm = Y[n_y*(f+1):, :]
    U_fm = U[n_u*(f+1):, :]

    # step 1, calculate the oblique projections
    # ------------------------------------------
    # Y_p = G_i Xd_p + Hd_i U_p
    # After the oblique projection, U_p component is eliminated,
    # without changing the Xd_p component:
    # Proj_perp_(U_p) Y_p = W1 O_i W2 = G_i Xd_p
    O_i = Y_f @ project_oblique(U_f, W_p)
    O_im = Y_fm @ project_oblique(U_fm, W_pp)

    # step 2, calculate the SVD of the weighted oblique projection
    # ------------------------------------------
    # given: W1 O_i W2 = G_i Xd_p
    # want to solve for G_i, but know product, and not Xd_p
    # so can only find Xd_p up to a similarity transformation
    W1 = np.eye(O_i.shape[0], dtype=numpy.float16)

    W2 = np.eye(O_i.shape[1], dtype=numpy.float16)
    U0, s0, VT0 = np.linalg.svd(W1 @ O_i @ W2)  # pylint: disable=unused-variable

    # step 3, determine the order by inspecting the singular
    # ------------------------------------------
    # values in S and partition the SVD accordingly to obtain U1, S1
    # print s0
    if order == -1:
        n_x = int(np.where(s0 / s0.max() > s_tol)[0][-1] + 1)  # Cupy doesn't see it as int, but as ndarray
    else:
        n_x = order
    # print("n_x", n_x)

    U1 = U0[:, :n_x]
    # S1 = np.matrix(np.diag(s0[:n_x]))
    # VT1 = VT0[:n_x, :n_x]

    # step 4, determine Gi and Gim
    # ------------------------------------------
    G_i = np.linalg.pinv(W1) @ U1 @ np.diag(np.sqrt(s0[:n_x]))
    G_im = G_i[:-n_y, :]  # check

    # step 5, determine Xd_ip and Xd_p
    # ------------------------------------------
    # only know Xd up to a similarity transformation
    Xd_i = np.linalg.pinv(G_i) @ O_i
    Xd_ip = np.linalg.pinv(G_im) @ O_im

    # step 6, solve the set of linear eqs
    # for A, B, C, D
    # ------------------------------------------
    Y_ii = Y[n_y*p:n_y*(p+1), :]
    U_ii = U[n_u*p:n_u*(p+1), :]

    a_mat = np.vstack([Xd_ip, Y_ii])
    b_mat = np.vstack([Xd_i, U_ii])

    # ss_mat = a_mat*b_mat.I
    ss_mat = a_mat @ np.linalg.pinv(b_mat)

    A_id = ss_mat[:n_x, :n_x]
    B_id = ss_mat[:n_x, n_x:]
    assert B_id.shape[0] == n_x
    assert B_id.shape[1] == n_u
    C_id = ss_mat[n_x:, :n_x]
    assert C_id.shape[0] == n_y
    assert C_id.shape[1] == n_x
    D_id = ss_mat[n_x:, n_x:]
    assert D_id.shape[0] == n_y
    assert D_id.shape[1] == n_u

    if np.linalg.matrix_rank(C_id) == n_x:
        T = np.linalg.pinv(C_id)  # try to make C identity, want it to look like state feedback
    else:
        T = np.eye(n_x, dtype=numpy.float16)

    Q_id = np.zeros((n_x, n_x), dtype=numpy.float16)
    R_id = np.zeros((n_y, n_y), dtype=numpy.float16)
    sys = sysid.StateSpaceDiscreteLinear(
        A=np.linalg.pinv(T) @ A_id @ T, B=np.linalg.pinv(T) @ B_id, C=C_id @ T, D=D_id,
        Q=Q_id, R=R_id, dt=dt)
    return sys


def nrms(data_fit, data_true):
    """
    Normalized root mean square error.
    See: https://nl.mathworks.com/help/ident/ref/goodnessoffit.html
    """

    # root_mean_squared_error = np.mean(np.linalg.norm(data_fit - data_true, axis=0), dtype=numpy.float16)  # RMSE
    # normalization_factor = 2 * np.linalg.norm(data_true - np.mean(data_true, axis=1), axis=0).max()
    # return (normalization_factor - root_mean_squared_error) / normalization_factor

    return (np.linalg.norm(data_true - data_fit))/(np.linalg.norm(data_true - np.mean(data_true)))


def symmetric_mean_absolute_percentage_error(a, b):
    """

    Calculates symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
        a (): ctual values.
        b (): Predicted values.

    Returns: sMAPE float %.

    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return 100.0 * np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mean_absolute_scaled_error(expected, predicted, naive_lags=1):
    """
    Mean Absolute Scaled Error (MASE) [1]: MAE / MAE_Naive_lag.
    [1] Hyndman RJ, Koehler AB. Another look at measures of forecast accuracy. International Journal of Forecasting.
    2006;22(4):679–688. 10.1016/j.ijforecast.2006.03.001
    :param expected: 1D or nD array of expected values
    :param predicted: 1D or nD array of predicted values
    :param naive_lags: Lags to scale for. Default 1 for non-stationary data.
    :return: 1D or nD Mean Absolute Scaled Error (MASE).
    """
    return mean_absolute_error(
        expected, predicted) / mean_absolute_error(expected, mimo_shift(expected, naive_lags, fill_value=expected[0]))

def mimo_shift(array, lags, fill_value=np.nan):
    """
    Shift the 1D or nD time-series by num steps. Returns the Naive-lag time-series.
    :param array: 1D or nD input array.
    :param lags: Steps to shift/lag.
    :param fill_value: Value to fill in the first lag steps that are empty.
    :return: 1D or nD naive-lag time-series.
    """
    result = np.empty_like(array)
    if lags > 0:
        result[:lags] = fill_value
        result[lags:] = array[:-lags]
    elif lags < 0:
        result[lags:] = fill_value
        result[:lags] = array[-lags:]
    else:
        result[:] = array
    return result


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    :param y_true: actual values
    :param y_pred: predicted values
    :return: MAPE float.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # In %


def index_of_agreement(validation, prediction):
    """
    Calculates Index Of Agreement (IOA).
    :param validation: actual values
    :param prediction: predicted values
    :return: IOA float.
    """
    return 1 - (np.sum((validation - prediction) ** 2)) / (np.sum((np.abs(prediction -
      np.mean(validation)) + np.abs(validation - np.mean(validation))) ** 2))


def prbs(n):
    """
    Pseudo random binary sequence.
    """
    return np.where(np.random.rand(n) > 0.5, 0, 1)
