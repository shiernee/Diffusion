import numpy as np
import copy as cp
import pandas as pd


class Parameters:
    def __init__(self, case):

        self.duration = None
        self.u0 = None
        self.dt = None
        self.nstep = None
        self.D = None

        # =============================================
        parameter_file = '../data/{}/param_template.csv'.format(case)
        param = pd.read_csv(parameter_file)
        self.duration = param['duration'].values[0]

        ic = InitialCondition(case)
        self.u0 = ic.get_u0()

        interpolated_spacing = param['interpolated_spacing_value'].values[0]
        self.D = param['D'].values[0]
        self.dt = self.compute_dt(interpolated_spacing, self.D)

        self.nstep = self.duration // self.dt

    def compute_dt(self, interpolated_spacing, D):
        self.dt = 0.4 * interpolated_spacing ** 2 / D
        return cp.copy(self.dt)


class InitialCondition:
    def __init__(self, case):

        self.u0 = None

        # ==================================================================
        coordinate_file = '../data/{}/coordinates.csv'.format(case)
        parameter_file = '../data/{}/param_template.csv'.format(case)

        coord = pd.read_csv(coordinate_file).values

        param = pd.read_csv(parameter_file)
        u0_method = param['u0_method'].values[0]

        self.set_u_initial(coord, u0_method)

    def set_u_initial(self, coord, u0_method):
        """

        :param coord:
        :param u0_method:
        :return:
        """
        r, phi, theta = self.xyz2sph(coord)
        if u0_method == 'cos_theta':
            self.u0 = np.cos(theta)
        elif u0_method == 'sin_square_theta_cos_phi':
            self.u0 = np.sin(theta) ** 2 * np.cos(phi)
        else:
            raise ValueError('u_initial is not set')
        return


    @staticmethod
    def xyz2sph(cart_coord):
        x, y, z = cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)

        phi = np.zeros_like(theta)
        idx = np.argwhere(x > 0.0).squeeze()
        phi[idx] = np.arctan(y[idx] / x[idx])

        idx = np.argwhere((x < 0.0) & (y >= 0)).squeeze()
        phi[idx] = np.arctan(y[idx] / x[idx]) + np.pi

        idx = np.argwhere((x < 0.0) & (y < 0)).squeeze()
        phi[idx] = np.arctan(y[idx] / x[idx]) - np.pi

        idx = np.argwhere((x == 0) & (y > 0)).squeeze()
        phi[idx] = np.pi / 2

        idx = np.argwhere((x == 0) & (y < 0)).squeeze()
        phi[idx] = - np.pi / 2

        idx = np.argwhere((x == 0.0) & (y == 0))
        phi[idx] = 0.0
        return r, phi, theta

    def get_u0(self):
        return self.u0.copy()

