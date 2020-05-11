from os import sys, path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))

import numpy as np
from scipy.spatial import distance
from Utils.Utils import xyz2r_phi_theta


class InterpolatedSpacing:

    # HK why you have the interpolated spacing method and then got 
    # HK value also, like that very confusing, do you use the 
    # HK scalar values at all?
    # def __init__(self, method, cart_coord, interpolated_spacing):
    def __init__(self, method, cart_coord, min_spacing): #HK
        """
        :param method:
        :param kwargs: cart_coord, interpolated_spacing
        """

        if method == 'avg_dx':
            spacing = self.interpolated_spacing_avg_dx(cart_coord)
        elif method == 'min_dist':
            spacing = self.interpolated_spacing_min_dist(cart_coord)
        elif method == 'scalar':
            spacing = min_spacing

        self.interpolated_spacing = min(spacing, min_spacing)

        self.dt = 0.1 * self.interpolated_spacing ** 2

    def interpolated_spacing_avg_dx(self, cart_coord):
        """

        :param cart_coord: cartesian coordinates
        :return:
        """
        r, phi, theta = xyz2r_phi_theta(cart_coord)
        no_pt = len(cart_coord)
        r = r.mean()
        cal_spacing = np.sqrt(4 * np.pi * r ** 2 / no_pt) / 2
        return cal_spacing

    def interpolated_spacing_min_dist(self, cart_coord):

        cal_spacing = distance.pdist(cart_coord).min()
        return cal_spacing


