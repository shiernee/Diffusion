#import sys
#sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Utils\src\\utils')

from Utils.src.utils.Utils import Utils
import pandas as pd
import numpy as np


class GeneratePoints:
    def __init__(self):
        return

    def generate_Niederreiter_datasets(self, no_pt, max_x, max_y):
        """

        :param filename: Niederreiter.csv
        :return: 2D array
        """
        filename = '../data/case2_NIEDERREITER2_DATASET/niederreiter2_02_10000.csv'
        dataframe_raw = pd.read_csv(filename, header=None)
        dataframe_raw = dataframe_raw.drop([0], axis=1)
        dataframe_raw.columns = ['x', 'y']

        # scaling
        dataframe_raw['x'] = dataframe_raw['x'] * max_x
        dataframe_raw['y'] = dataframe_raw['y'] * max_y

        dataframe = dataframe_raw.head(no_pt)
        coord = np.zeros([no_pt, 3])
        coord[:, 0], coord[:, 1] = dataframe['x'].values, dataframe['y'].values

        return coord

    def generate_point_regular_grid(self, no_pt, max_x, max_y):
        """

        :param dx: flaot
        :param max_x: flaot
        :param max_y: flaot
        :return: 2D array
        """
        x = np.linspace(0, max_x, np.sqrt(no_pt))
        y = np.linspace(0, max_y, np.sqrt(no_pt))
        X, Y = np.meshgrid(x, y)
        coord = np.zeros([len(X.flatten()), 3])
        coord[:, 0] = X.flatten()
        coord[:, 1] = Y.flatten()

        return coord

    def generate_points_2D(self, no_pt, max_x, max_y):
        """

        :param no_pt: int
        :param max_x: float
        :param max_y: float
        :return: 2D array
        """
        # no_pt = int(max_x / dx + 1) * int(max_x / dx + 1)
        x = np.random.rand(no_pt, ) * max_x
        y = np.random.rand(no_pt, ) * max_y
        coord = np.zeros([no_pt, 3])
        coord[:, 0] = x
        coord[:, 1] = y

        return coord

    def generate_points_sphere(self, no_pt, sph_radius):
        """

        :param n_point: int
        :param sph_radius: float
        :return: 2D array
        """
        ut = Utils()
        phi = np.random.uniform(-np.pi, np.pi, no_pt)
        theta = np.random.uniform(0, np.pi, no_pt)
        radius = np.ones([no_pt, ]) * sph_radius

        sph_coord = np.zeros([no_pt, 3])
        sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2] = radius, phi, theta
        x, y, z = ut.sph2xyz(sph_coord)

        cart_coord = np.zeros([no_pt, 3])
        cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2] = x, y, z

        return cart_coord, sph_coord


if __name__ == '__main__':
    gen_pt = GeneratePoints()
    cart_coord, sph_coord = gen_pt.generate_points_sphere(no_pt=1001, sph_radius=1)
    cart_coord = gen_pt.generate_points_2D(no_pt=1001, max_x=55, max_y=55)
    cart_coord = gen_pt.generate_point_regular_grid(no_pt=1001, max_x=55, max_y=55)
    cart_coord = gen_pt.generate_Niederreiter_datasets(no_pt=1001, max_x=55, max_y=55)

