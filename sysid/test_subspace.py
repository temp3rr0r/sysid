"""
Unit testing.
"""
import unittest
import matplotlib.pyplot as plt
import numpy as np

# import sysid
# import sysid.subspace
# import sysid.subspace_bak as subspace_bak
# import sysid.ss_bak as ss_bak

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
        import sysid
        import sysid.subspace
        y = np.random.rand(3, 100)
        Y = sysid.subspace.block_hankel(y, 5)
        self.assertEqual(Y.shape, (15, 95))

    def test_block_hankel_long(self):
        """
        Block hankel function.
        """
        import sysid
        import sysid.subspace
        y = np.random.rand(200, 3000)
        Y = sysid.subspace.block_hankel(y, 5)
        self.assertEqual(Y.shape, (1000, 2995))

    def test_project(self):
        import sysid
        import sysid.subspace
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        Y = sysid.subspace.project(A)
        print(Y)
        # self.assertEqual(Y, np.array([[1.78125, 1.4375, 1.09375], [0.875, 1., 1.125],  [-0.28125, 0.0625, 0.40625]]))

    def test_project2(self):
        import sysid.subspace_bak as subspace_bak
        import sysid.ss_bak as ss_bak
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        Y = subspace_bak.project(A)
        print(Y)
        # self.assertEqual(Y, np.array([[1.78125, 1.4375, 1.09375], [0.875, 1., 1.125],  [-0.28125, 0.0625, 0.40625]]))

    def test_project_perp(self):
        import sysid
        import sysid.subspace
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        Y = sysid.subspace.project_perp(A)
        print(Y)
        # [[-0.78125 - 1.4375 - 1.09375]
        #  [-0.875    0. - 1.125]
        # [0.28125 - 0.0625 0.59375]]

    def test_project_perp2(self):
        import sysid.subspace_bak as subspace_bak
        import sysid.ss_bak as ss_bak
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        Y = subspace_bak.project_perp(A)
        print(Y)
        # [[-0.78125 - 1.4375 - 1.09375]
        #  [-0.875    0. - 1.125]
        # [0.28125 - 0.0625 0.59375]]

    def test_project_oblique(self):
        import sysid
        import sysid.subspace
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        B = np.array([[3, 2, 3], [3, 2, 2], [7, 8, 7]])
        print(B)
        Y = sysid.subspace.project_oblique(A, B)
        print(Y)
        # [[ 1.00000000e+00  8.88178420e-16  2.22044605e-15]
        #  [ 1.22124533e-15  1.00000000e+00  1.22124533e-15]
        #  [ 5.37764278e-16 -9.71445147e-16  1.00000000e+00]]

    def test_project_oblique2(self):
        import sysid.subspace_bak as subspace_bak
        import sysid.ss_bak as ss_bak
        A = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        print(A)
        B = np.array([[3, 2, 3], [3, 2, 2], [7, 8, 7]])
        print(B)
        Y = subspace_bak.project_oblique(A, B)
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
        import sysid
        import sysid.subspace
        ss1 = sysid.ss.StateSpaceDiscreteLinear(
            A=0.9, B=0.5, C=1, D=0, Q=0.01, R=0.01, dt=0.1)
        # TODO
        np.random.seed(1234)
        prbs1 = sysid.subspace.prbs(1000)
        def f_prbs(t, x, i):
            return prbs1[i]
        tf = 10
        data = ss1.simulate(f_u=f_prbs, x0=np.array(0), tf=tf)

    def test_subspace_simulate2(self):
        import sysid.subspace_bak as subspace_bak
        import sysid.ss_bak as ss_bak
        ss1 = ss_bak.StateSpaceDiscreteLinear(
            A=0.9, B=0.5, C=1, D=0, Q=0.01, R=0.01, dt=0.1)
        np.random.seed(1234)
        prbs1 = subspace_bak.prbs(1000)
        def f_prbs(t, x, i):
            return prbs1[i]
        tf = 10
        data = ss1.simulate(f_u=f_prbs, x0=np.matrix(0), tf=tf)

    def test_subspace_det_algo1_siso(self):
        """
        Subspace deterministic algorithm (SISO).
        """
        import sysid
        import sysid.subspace
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

    def test_subspace_det_algo1_siso2(self):
        import sysid.subspace_bak as subspace_bak
        import sysid.ss_bak as ss_bak

        ss1 = ss_bak.StateSpaceDiscreteLinear(
            A=0.9, B=0.5, C=1, D=0, Q=0.01, R=0.01, dt=0.1)
        np.random.seed(1234)
        prbs1 = subspace_bak.prbs(1000)
        def f_prbs(t, x, i):
            return prbs1[i]
        tf = 10
        data = ss1.simulate(f_u=f_prbs, x0=np.array(0), tf=tf)
        ss1_id = subspace_bak.subspace_det_algo1(
            y=data.y, u=data.u,
            f=5, p=5, s_tol=1e-1, dt=ss1.dt)
        data_id = ss1_id.simulate(f_u=f_prbs, x0=0, tf=tf)
        nrms = subspace_bak.nrms(data_id.y, data.y)
        self.assertGreater(nrms, 0.9)

    def test_subspace_det_algo1_mimo(self):
        """
        Subspace deterministic algorithm (MIMO).
        """
        import sysid
        import sysid.subspace
        ss2 = sysid.StateSpaceDiscreteLinear(
            A=np.matrix([[0, 0.1, 0.2],
                         [0.2, 0.3, 0.4],
                         [0.4, 0.3, 0.2]]),
            B=np.matrix([[1, 0],
                         [0, 1],
                         [0, -1]]),
            C=np.matrix([[1, 0, 0],
                         [0, 1, 0]]),
            D=np.matrix([[0, 0],
                         [0, 0]]),
            Q=np.diag([0.01, 0.01, 0.01]), R=np.diag([0.01, 0.01]), dt=0.1)
        np.random.seed(1234)
        prbs1 = sysid.prbs(1000)
        prbs2 = sysid.prbs(1000)

        def f_prbs_2d(t, x, i):
            "input function"
            #pylint: disable=unused-argument
            i = i % 1000
            return 2*np.matrix([prbs1[i]-0.5, prbs2[i]-0.5]).T
        tf = 8
        data = ss2.simulate(
            f_u=f_prbs_2d, x0=np.matrix([0, 0, 0]).T, tf=tf)
        ss2_id = sysid.subspace_det_algo1(
            y=data.y, u=data.u,
            f=5, p=5, s_tol=0.1, dt=ss2.dt)
        data_id = ss2_id.simulate(
            f_u=f_prbs_2d,
            x0=np.matrix(np.zeros(ss2_id.A.shape[0])).T, tf=tf)
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


if __name__ == "__main__":
    unittest.main()

# vim: set et ft=python fenc=utf-8 ff=unix sts=4 sw=4 ts=4 :
