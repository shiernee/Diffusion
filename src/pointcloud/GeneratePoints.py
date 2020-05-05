import pandas as pd
import numpy as np
import os
from os import path

np.random.seed(23873)


class GenerateNiederreiterDatasets:
    def __init__(self, no_pt, max_x, max_y):
        """
        :param n_point: int
        :param max_x: float
        :param max_y: float
        :return: void
        """
        self.cart_coord = None

        self.no_pt = no_pt
        self.max_x = max_x
        self.max_y = max_y

        # ===================================
        parent_file = path.dirname(path.dirname(path.abspath(__file__)))
        filename = os.path.join(parent_file, "data", "niederreiter2_02_10000.csv")
        dataframe_raw = pd.read_csv(filename)
        dataframe_raw.columns = ['x', 'y']

        # scaling
        dataframe_raw['x'] = dataframe_raw['x'] * self.max_x
        dataframe_raw['y'] = dataframe_raw['y'] * self.max_y

        dataframe = dataframe_raw.head(self.no_pt)
        self.cart_coord = np.zeros([self.no_pt, 3])
        self.cart_coord[:, 0], self.cart_coord[:, 1] = dataframe['x'].values, dataframe['y'].values

        return

    def create_README_file(self, path):
        filename = os.path.join(path, "README.txt")
        with open(filename, mode='w', newline='') as csv_file:
            csv_file.write('Niederreiter\n')
            csv_file.write('no_pt:{}\n'.format(self.no_pt))
            csv_file.write('max_x:{}\n'.format(self.max_x))
            csv_file.write('max_y:{}\n'.format(self.max_y))
        print('{}'.format(filename))


class GenerateRegular2DGrid:
    def __init__(self, no_pt, max_x, max_y):
        """

        :param no_pt: int
        :param max_x: float
        :param max_y: float
        :return: void
        """
        self.cart_coord = None

        self.no_pt = no_pt
        self.max_x = max_x
        self.max_y = max_y

        ut = Utils()

        # ===================================
        len_x, len_y = int(np.sqrt(no_pt)), int(np.sqrt(no_pt))
        x = np.linspace(0, max_x, len_x)
        y = np.linspace(0, max_y, len_y)
        X, Y = np.meshgrid(x, y)
        self.cart_coord = np.zeros([len(X.flatten()), 3])
        self.cart_coord[:, 0] = X.flatten()
        self.cart_coord[:, 1] = Y.flatten()

        return

    def create_README_file(self, path):
        filename = os.path.join(path, "README.txt")
        with open(filename, mode='w', newline='') as csv_file:
            csv_file.write('Regular2DGrid\n')
            csv_file.write('no_pt:{}\n'.format(self.no_pt))
            csv_file.write('max_x:{}\n'.format(self.max_x))
            csv_file.write('max_y:{}\n'.format(self.max_y))
        print('{}'.format(filename))


class GenerateScatter2DPoints:
    def __init__(self, no_pt, max_x, max_y):
        """
        :param no_pt: int
        :param max_x: float
        :param max_y: float
        :return: void
        """
        self.cart_coord = None
        self.no_pt = no_pt
        self.max_x = max_x
        self.max_y = max_y

        ut = Utils()

        # ===================================
        x = np.random.rand(no_pt, ) * max_x
        y = np.random.rand(no_pt, ) * max_y
        self.cart_coord = np.zeros([no_pt, 3])
        self.cart_coord[:, 0] = x
        self.cart_coord[:, 1] = y

        return

    def create_README_file(self, path):
        filename = os.path.join(path, "README.txt")
        with open(filename, mode='w', newline='') as csv_file:
            csv_file.write('Scatter2DGrid\n')
            csv_file.write('no_pt:{}\n'.format(self.no_pt))
            csv_file.write('max_x:{}\n'.format(self.max_x))
            csv_file.write('max_y:{}\n'.format(self.max_y))
        print('{}'.format(filename))


class GenerateSphPoints:
    def __init__(self, no_pt, sph_radius):
        """
        :param n_point: int
        :param sph_radius: float
        :return: void
       """
        self.cart_coord = None

        self.no_pt = no_pt
        self.sph_radius = sph_radius

        ut = Utils()

        # ===========================================
        phi = np.random.uniform(-np.pi, np.pi, no_pt)
        theta = np.random.uniform(0, np.pi, no_pt)
        radius = np.ones([no_pt, ]) * sph_radius

        sph_coord = np.zeros([no_pt, 3])
        sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2] = radius, phi, theta

        x, y, z = ut.sph2xyz(sph_coord)

        self.cart_coord = np.zeros([no_pt, 3])
        self.cart_coord[:, 0], self.cart_coord[:, 1], self.cart_coord[:, 2] = x, y, z
        return

    def create_README_file(self, path):
        filename = os.path.join(path, "README.txt")
        with open(filename, mode='w', newline='') as csv_file:
            csv_file.write('sphere\n')
            csv_file.write('no_pt:{}\n'.format(self.no_pt))
            csv_file.write('sphere_radius:{}\n'.format(self.sph_radius))
        print('{}'.format(filename))


class Utils:
    def __init__(self):
        return

    @staticmethod
    def sph2xyz(sph_coord):
        sph_radius, phi, theta = sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2]
        x = sph_radius * np.sin(theta) * np.cos(phi)
        y = sph_radius * np.sin(theta) * np.sin(phi)
        z = sph_radius * np.cos(theta)
        return x, y, z

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




