"""
Unit testing.
"""
import unittest
import matplotlib.pyplot as plt
import numpy as np

import sysid
import time
import sysid.subspace

# pylint: disable=invalid-name, no-self-use

ENABLE_PLOTTING = False


class TestSubspace(unittest.TestCase):
    """
    Unit testing.
    """

    def test_block_hankel(self):
        """
        Block hankel function.
        """
        y = np.random.rand(3, 100)
        Y = sysid.subspace.block_hankel(y, 5)
        self.assertEqual(Y.shape, (15, 95))

    def test_block_hankel_long(self):
        """
        Block hankel function.
        """
        y = np.random.rand(200, 3000)
        Y = sysid.subspace.block_hankel(y, 5)
        self.assertEqual(Y.shape, (1000, 2995))

    def test_project(self):
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        Y = sysid.subspace.project(A)
        print(Y)
        # self.assertEqual(Y, np.array([[1.78125, 1.4375, 1.09375], [0.875, 1., 1.125],  [-0.28125, 0.0625, 0.40625]]))


    def test_project_perp(self):
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        Y = sysid.subspace.project_perp(A)
        print(Y)
        # [[-0.78125 - 1.4375 - 1.09375]
        #  [-0.875    0. - 1.125]
        # [0.28125 - 0.0625 0.59375]]

    def test_project_oblique(self):
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        B = np.array([[3, 2, 3], [3, 2, 2], [7, 8, 7]])
        print(B)
        Y = sysid.subspace.project_oblique(A, B)
        print(Y)
        # [[ 1.00000000e+00  8.88178420e-16  2.22044605e-15]
        #  [ 1.22124533e-15  1.00000000e+00  1.22124533e-15]
        #  [ 5.37764278e-16 -9.71445147e-16  1.00000000e+00]]

    def test_svd(self):
        """
        Block hankel function.
        """
        y = np.random.rand(100, 0)

    def test_subspace_simulate(self):
        # ss1 = sysid.ss.StateSpaceDiscreteLinear(
        #     A=0.9, B=0.5, C=1, D=0, Q=0.01, R=0.01, dt=0.1)
        ss1 = sysid.StateSpaceDiscreteLinear(
            A=np.array([[0.9]]),
            B=np.array([[0.5]]),
            C=np.array([[1]]),
            D=np.array([[0]]),
            Q=np.diag([0.01]), R=np.diag([0.01]), dt=0.1)

        np.random.seed(1234)
        # prbs1 = np.array(np.matrix(sysid.subspace.prbs(1000)))
        prbs1 = sysid.subspace.prbs(1000)
        def f_prbs(t, x, i):
            return prbs1[i]
        tf = 10
        data = ss1.simulate(f_u=f_prbs,
                            # x0=np.array(0),
                            x0=np.array([[0]]).T,
                            tf=tf)

    def test_subspace_det_algo1_siso(self):
        """
        Subspace deterministic algorithm (SISO).
        """
        ss1 = sysid.StateSpaceDiscreteLinear(
            A=0.9, B=0.5, C=1, D=0, Q=0.01, R=0.01, dt=0.1)

        np.random.seed(1234)
        prbs1 = sysid.prbs(1000)

        def f_prbs(t, x, i):
            "input function"
            # pylint: disable=unused-argument, unused-variable
            return prbs1[i]

        tf = 10
        data = ss1.simulate(f_u=f_prbs, x0=np.matrix(0), tf=tf)
        ss1_id = sysid.subspace_det_algo1(
            y=data.y, u=data.u,
            f=5, p=5, s_tol=1e-1, dt=ss1.dt)
        data_id = ss1_id.simulate(f_u=f_prbs, x0=0, tf=tf)
        nrms = sysid.subspace.nrms(data_id.y, data.y)
        self.assertGreater(nrms, 0.9)

        if ENABLE_PLOTTING:
            plt.plot(data_id.t.T, data_id.x.T, label='id')
            plt.plot(data.t.T, data.x.T, label='true')
            plt.legend()
            plt.grid()


    def test_subspace_det_algo1_mimo(self):
        """
        Subspace deterministic algorithm (MIMO).
        """
        ss2 = sysid.StateSpaceDiscreteLinear(
            A=np.array([[0, 0.1, 0.2],
                         [0.2, 0.3, 0.4],
                         [0.4, 0.3, 0.2]]),
            B=np.array([[1, 0],
                         [0, 1],
                         [0, -1]]),
            C=np.array([[1, 0, 0],
                         [0, 1, 0]]),
            D=np.array([[0, 0],
                         [0, 0]]),
            Q=np.diag([0.01, 0.01, 0.01]), R=np.diag([0.01, 0.01]), dt=0.1)
        np.random.seed(1234)
        prbs1 = sysid.prbs(1000)
        prbs2 = sysid.prbs(1000)

        def f_prbs_2d(t, x, i):
            "input function"
            #pylint: disable=unused-argument
            i = i % 1000
            return 2 * np.array([[prbs1[i]-0.5], [prbs2[i]-0.5]])
        tf = 8
        data = ss2.simulate(
            f_u=f_prbs_2d,
            x0 =np.array([[0, 0, 0]]).T,
            tf=tf)
        ss2_id = sysid.subspace_det_algo1(
            y=data.y, u=data.u,
            f=5, p=5, s_tol=0.1, dt=ss2.dt)
        data_id = ss2_id.simulate(
            f_u=f_prbs_2d,
            # x0=np.array(np.matrix(np.zeros(ss2_id.A.shape[0])).T),
            x0=np.array([np.zeros(ss2_id.A.shape[0])]).T,
            tf=tf)

        nrms = sysid.nrms(data_id.y, data.y)
        self.assertGreater(nrms, 0.9)

        if ENABLE_PLOTTING:
            for i in range(2):
                plt.figure()
                plt.plot(data_id.t.T, data_id.y[i, :].T,
                         label='$y_{:d}$ true'.format(i))
                plt.plot(data.t.T, data.y[i, :].T,
                         label='$y_{:d}$ id'.format(i))
                plt.legend()
                plt.grid()

    def test_subspace_det_algo1_mimo2(self):
        tf = 36 * 8
        dt = 1
        in_size = 5
        out_size = 2
        data_u = np.random.randn(in_size, tf)
        data_y = np.random.randn(out_size, tf)
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
        print("--- Serial:\t\t{} seconds".format(time.time() - start_time))
        data3_id = ss3_id.simulate(
            f_u=f_prbs_4d,
            x0=np.array([np.zeros(ss3_id.A.shape[0])]).T,
            tf=tf)
        print('fit {:f}%'.format(100 * sysid.subspace.nrms(data3_id.y, data_y[:, -1:])))


    def test_subspace_det_algo1_mimo3(self):
        tf = 365 * 8
        dt = 1
        in_size = 50
        out_size = 5
        data_u = np.random.randn(in_size, tf)
        data_y = np.random.randn(out_size, tf)
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
        print("--- Serial:\t\t{} seconds".format(time.time() - start_time))
        data3_id = ss3_id.simulate(
            f_u=f_prbs_4d,
            x0=np.array([np.zeros(ss3_id.A.shape[0])]).T,
            tf=tf)
        print('fit {:f}%'.format(100 * sysid.subspace.nrms(data3_id.y, data_y[:, -1:])))

if __name__ == "__main__":
    unittest.main()

# vim: set et ft=python fenc=utf-8 ff=unix sts=4 sw=4 ts=4 :
