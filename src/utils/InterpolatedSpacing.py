#import sys
#sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Utils\src\\utils')

import numpy as np
from scipy.spatial import distance
from Utils.src.utils.Utils import Utils


class InterpolatedSpacing:
    def __init__(self, method, cart_coord, interpolated_spacing):
        """

        :param method:
        :param kwargs: cart_coord, interpolated_spacing
        """

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
        ut = Utils()
        r, phi, theta = ut.xyz2sph(cart_coord)
        no_pt = len(cart_coord)
        r = r.mean()
        return np.sqrt(4 * np.pi * r ** 2 / no_pt)

    def interpolated_spacing_min_dist(self, cart_coord):
        return distance.pdist(cart_coord).min()

