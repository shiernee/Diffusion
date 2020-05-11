from os import sys, path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from scipy.interpolate import Rbf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils.Utils import xyz2r_phi_theta


class ExactInterpolator4test:

    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        return

    # =================================================
    def exact_formula(self, theta, phi):
        return np.sin(theta)*np.sin(theta)*np.cos(phi)

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]

        local_grid = local_grid.reshape([-1, 3])

        r, phi, theta = xyz2r_phi_theta(local_grid)
        intp_val = self.exact_formula(theta, phi)
        intp_val = intp_val.reshape(grid_shape)

        return intp_val  # see variables.py lines 50-70
    # =================================================



