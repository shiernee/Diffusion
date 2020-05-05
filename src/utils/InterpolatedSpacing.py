import numpy as np
from scipy.spatial import distance


class InterpolatedSpacing:
    def __init__(self, method, cart_coord, interpolated_spacing):
        """

        :param method:
        :param kwargs: cart_coord, interpolated_spacing
        """

        self.min_spacing = 0.05 #HK min spacing

        if method == 'avg_dx':
            self.interpolated_spacing = self.interpolated_spacing_avg_dx(cart_coord)
        elif method == 'min_dist':
            self.interpolated_spacing = self.interpolated_spacing_min_dist(cart_coord)
        elif method == 'scalar':
            self.interpolated_spacing = interpolated_spacing

        self.dt = 0.1 * self.interpolated_spacing ** 2

    def interpolated_spacing_avg_dx(self, cart_coord):
        """

        :param cart_coord: cartesian coordinates
        :return:
        """
        r, phi, theta = self.xyz2sph(cart_coord)
        no_pt = len(cart_coord)
        r = r.mean()
        return np.sqrt(4 * np.pi * r ** 2 / no_pt) / 2

    def interpolated_spacing_min_dist(self, cart_coord):

        # HK return distance.pdist(cart_coord).min()
        # HK cal_spacing = distance.pdist(cart_coord).min()
        # HK return min(cal_spacing,self.min_spacing)
        cal_spacing = distance.pdist(cart_coord).min()
        return min(cal_spacing,self.min_spacing)

        # HK return distance.pdist(cart_coord).min()

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
