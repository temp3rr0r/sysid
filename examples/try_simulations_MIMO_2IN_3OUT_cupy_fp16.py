import sysid
import sysid.subspace

import time
import pandas as pd

# import numpy as np
import cupy as np
import numpy

simulation_example = False

if simulation_example:
    ss3 = sysid.StateSpaceDiscreteLinear(
        A=np.array([[0, 0.01, 0.2, 0.4], [0.1, 0.2, 0.2, 0.3], [0.11, 0.12, 0.21, 0.13], [0.11, 0.12, 0.21, 0.13]], dtype=numpy.float16),  # X x X
        B=np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=numpy.float16),  # X x u
        C=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=numpy.float16),  # y x x
        D=np.array([[0, 0], [0, 0], [0, 0]], dtype=numpy.float16),  # y x u
        Q=np.diag([0.2, 0.1, 0.1, 0.1]),  #  X x X
        R=np.diag([0.04, 0.04, 0.04]),  # R? y x y
        dt=0.1)

    np.random.seed(1234)
    prbs1 = sysid.prbs(1000)
    prbs2 = sysid.prbs(1000)
    prbs3 = sysid.prbs(1000)
    def f_prbs_3d(t, x, i):
        i = i % 1000
        return 2 * np.array([[prbs1[i]-0.5], [prbs2[i]-0.5]], dtype=numpy.float16)

    tf = 10
    data3 = ss3.simulate(
        f_u=f_prbs_3d,
        x0=np.array([[0, 0, 0, 0]], dtype=numpy.float16).T,
        tf=tf)

    print(data3.u.shape, data3.y.shape, "shapes")

    ss3_id = sysid.subspace_det_algo1(y=data3.y, u=data3.u,
        f=5, p=5, s_tol=0.2, dt=ss3.dt)
    data3_id = ss3_id.simulate(
        f_u=f_prbs_3d,
        x0=np.array([np.zeros(ss3_id.A.shape[0])], dtype=numpy.float16).T,
        tf=tf)

    print(data3.u.shape)
    print(data3.x.shape)
    print(data3.y.shape)

tf = 2615  # 365 * 5
dt = 1

data_u = np.random.randn(10, tf)  # 40 * 45
data_y = np.random.randn(4, tf)  # 40 * 45
print("data_u.shape: {}, data_y.shape: {}".format(data_u.shape, data_y.shape))
print("MIMO [{} IN, {} OUT], {} time-steps.".format(data_u.shape[0], data_y.shape[0], data_u.shape[1]))

def f_prbs_4d(t, x, i):
    return np.array([data_u[:, i]], dtype=numpy.float16).T

start_time = time.time()  # Serial
ss3_id = sysid.subspace_det_algo1(y=data_y, u=data_u,
    f=5,  # 5 Forward steps
    p=5,  # 5 Backward steps
    s_tol=0.01,  # 0.2
    dt=dt,
    order=-1)
print("--- GPU Execution time:\t\t{} seconds".format(time.time() - start_time))

data3_id = ss3_id.simulate(
    f_u=f_prbs_4d,
    x0=np.array([np.zeros(ss3_id.A.shape[0])], dtype=numpy.float16).T,
    tf=tf)

print('NRMSE: {}'.format(sysid.subspace_cupy_fp16.nrms(data3_id.y, data_y[:, 1:])))
print('SMAPE: {}%'.format(sysid.subspace_cupy_fp16.symmetric_mean_absolute_percentage_error(data3_id.y, data_y[:, 1:])))
