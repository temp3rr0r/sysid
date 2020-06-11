import sysid
import sysid.subspace

import time
import pandas as pd

# import numpy as np
import cupy as np

simulation_example = False

if simulation_example:
    # 4 internal states MIMO [2 IN, 3 OUT]
    ss3 = sysid.StateSpaceDiscreteLinear(
        A=np.array([[0, 0.01, 0.2, 0.4], [0.1, 0.2, 0.2, 0.3], [0.11, 0.12, 0.21, 0.13], [0.11, 0.12, 0.21, 0.13]]),  # X x X
        B=np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),  # X x u
        C=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),  # y x x
        D=np.array([[0, 0], [0, 0], [0, 0]]),  # y x u
        Q=np.diag([0.2, 0.1, 0.1, 0.1]),  #  X x X
        R=np.diag([0.04, 0.04, 0.04]),  # R? y x y
        dt=0.1)

    np.random.seed(1234)
    prbs1 = sysid.prbs(1000)
    prbs2 = sysid.prbs(1000)
    prbs3 = sysid.prbs(1000)
    def f_prbs_3d(t, x, i):
        i = i % 1000
        return 2 * np.array([[prbs1[i]-0.5], [prbs2[i]-0.5]])

    tf = 10
    data3 = ss3.simulate(
        f_u=f_prbs_3d,
        x0=np.array([[0, 0, 0, 0]]).T,
        tf=tf)

    print(data3.u.shape, data3.y.shape, "shapes")

    ss3_id = sysid.subspace_det_algo1(y=data3.y, u=data3.u,
        f=5, p=5, s_tol=0.2, dt=ss3.dt)
    data3_id = ss3_id.simulate(
        f_u=f_prbs_3d,
        x0=np.array([np.zeros(ss3_id.A.shape[0])]).T,
        tf=tf)

    print(data3.u.shape)
    print(data3.x.shape)
    print(data3.y.shape)

tf = 365 * 5  # 365 * 5
dt = 1
plot_stuff = False

# TODO: Cupy fp16 works?
data_u = np.random.randn(40 * 45, tf)  # 40 * 45
data_y = np.random.randn(40 * 45, tf)  # 40 * 45
print("data_u.shape: {}, data_y.shape: {}".format(data_u.shape, data_y.shape))
print("MIMO [{} IN, {} OUT], {} time-steps.".format(data_u.shape[0], data_y.shape[0], data_u.shape[1]))

def f_prbs_4d(t, x, i):
    return np.array([data_u[:, i]]).T

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
    x0=np.array([np.zeros(ss3_id.A.shape[0])]).T,
    tf=tf)

# if plot_stuff:
#     for i in range(3):
#         pl.figure(figsize=(15,5))
#         pl.plot(data3_id.t.T, data3_id.y[i,:].T,
#                 label='$y_{:d}$ true'.format(i))
#         pl.plot(data3_id.t.T,
#                 np.matrix(data_y[i,:-1]).T,
#                 label='$y_{:d}$ id'.format(i))
#         pl.legend()
#         pl.grid()

print('fit {}%'.format(100*sysid.subspace.nrms(data3_id.y, data_y[:, -1:])))
