import numpy as np
import sysid

# # 4 internal states MIMO [2 IN, 2 OUT]
# # ss3 = sysid.StateSpaceDiscreteLinear(
# #     A=np.matrix([[0,0.01, 0.2, 0.4],[0.1, 0.2, 0.2,0.3],[0.11, 0.12, 0.21,0.13],[0.11, 0.12, 0.21,0.13]]),
# #     B=np.matrix([[1,0],[0,1],[1,0],[0,1]]),
# #     C=np.matrix([[1,0, 0, 0],[0,1,0,0]]),
# #     D=np.matrix([[0,0],[0,0]]),
# #     Q=np.diag([0.2,0.1,0.1,0.1]), # Q == K?
# #     R=np.diag([0.04,0.04]), # R?
# #     dt=0.1)
# # 4 internal states MIMO [3 IN, 2 OUT]
# ss3 = sysid.StateSpaceDiscreteLinear(
#     A=np.matrix([[0, 0.01, 0.2, 0.4], [0.1, 0.2, 0.2, 0.3], [0.11, 0.12, 0.21, 0.13], [0.11, 0.12, 0.21, 0.13]]),  # X x X
#     B = np.matrix([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]),  # X x u
#     C = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]]),  # y x x
#     D = np.matrix([[0, 0, 0], [0, 0, 0]]),  # y x u
#     Q = np.diag([0.2, 0.1, 0.1, 0.1]),  #  X x X
#     R = np.diag([0.04, 0.04]),  # R? y x y?
#     dt=0.1)
#
# np.random.seed(1234)
# prbs1 = sysid.prbs(1000)
# prbs2 = sysid.prbs(1000)
# prbs3 = sysid.prbs(1000)
# def f_prbs_3d(t, x, i):
#     i = i%1000
#     return 2*np.matrix([prbs1[i]-0.5, prbs2[i]-0.5, prbs3[i]-0.5]).T
#
# tf = 10
# data3 = ss3.simulate(
#     f_u=f_prbs_3d,
#     x0=np.matrix([0,0,0,0]).T,
#     tf=tf)
#
# print(data3.u.shape, data3.y.shape, "shapes")
#
# ss3_id = sysid.subspace_det_algo1(y=data3.y, u=data3.u,
#     f=5, p=5, s_tol=0.2, dt=ss3.dt)
# data3_id = ss3_id.simulate(
#     f_u=f_prbs_3d,
#     x0=np.matrix(np.zeros(ss3_id.A.shape[0])).T, tf=tf)
# ss3_id
#
# print(data3.u.shape)
# print(data3.x.shape)
# print(data3.y.shape)


# 4 internal states MIMO [2 IN, 3 OUT]
ss3 = sysid.StateSpaceDiscreteLinear(
    A=np.matrix([[0, 0.01, 0.2, 0.4], [0.1, 0.2, 0.2, 0.3], [0.11, 0.12, 0.21, 0.13], [0.11, 0.12, 0.21, 0.13]]),  # X x X
    B=np.matrix([[1, 0], [0, 1], [1, 0], [0, 1]]),  # X x u
    C=np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),  # y x x
    D=np.matrix([[0, 0], [0, 0], [0, 0]]),  # y x u
    Q=np.diag([0.2, 0.1, 0.1, 0.1]),  #  X x X
    R=np.diag([0.04, 0.04, 0.04]),  # R? y x y
    dt=0.1)

np.random.seed(1234)
prbs1 = sysid.prbs(1000)
prbs2 = sysid.prbs(1000)
prbs3 = sysid.prbs(1000)
def f_prbs_3d(t, x, i):
    i = i%1000
    return 2*np.matrix([prbs1[i]-0.5, prbs2[i]-0.5]).T

tf = 10
data3 = ss3.simulate(
    f_u=f_prbs_3d,
    x0=np.matrix([0,0,0,0]).T,
    tf=tf)

print(data3.u.shape, data3.y.shape, "shapes")

ss3_id = sysid.subspace_det_algo1(y=data3.y, u=data3.u,
    f=5, p=5, s_tol=0.2, dt=ss3.dt)
data3_id = ss3_id.simulate(
    f_u=f_prbs_3d,
    x0=np.matrix(np.zeros(ss3_id.A.shape[0])).T, tf=tf)

print(data3.u.shape)
print(data3.x.shape)
print(data3.y.shape)

# 4 internal states MIMO [2 IN, 3 OUT] np.array
ss3 = sysid.StateSpaceDiscreteLinear2(
    A=np.array([[0, 0.01, 0.2, 0.4], [0.1, 0.2, 0.2, 0.3], [0.11, 0.12, 0.21, 0.13], [0.11, 0.12, 0.21, 0.13]]),  # X x X
    B=np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),  # X x u
    C=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),  # y x x
    D=np.array([[0, 0], [0, 0], [0, 0]]),  # y x u
    Q=np.diag([0.2, 0.1, 0.1, 0.1]),  #  X x X
    R=np.diag([0.04, 0.04, 0.04]),  # R? y x y
    dt=0.1)

def f_prbs_3d(t, x, i):
    i = i%1000
    return np.array(2*np.matrix([prbs1[i]-0.5, prbs2[i]-0.5]).T)

tf = 10
data3 = ss3.simulate2(
    f_u=f_prbs_3d,
    x0=np.matrix([0,0,0,0]).T,
    tf=tf)

print(data3.u.shape, data3.y.shape, "shapes")

ss3_id = sysid.subspace_det_algo1(y=data3.y, u=data3.u,
    f=5, p=5, s_tol=0.2, dt=ss3.dt)
data3_id = ss3_id.simulate(
    f_u=f_prbs_3d,
    x0=np.matrix(np.zeros(ss3_id.A.shape[0])).T, tf=tf)

print(data3.u.shape)
print(data3.x.shape)
print(data3.y.shape)