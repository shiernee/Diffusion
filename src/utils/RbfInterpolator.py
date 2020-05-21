from os import sys, path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from scipy.interpolate import Rbf, NearestNDInterpolator, LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils.Utils import xyz2r_phi_theta
from src.utils.IDWInterpolator import IDWInterpolator


class RbfInterpolatorSphCubic:
    def __init__(self, point_cloud):
        print('RbfInterpolatorSphCubic')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        x, y, z = self.point_cloud.coord[nn_indices, 0], self.point_cloud.coord[nn_indices, 1], \
                  self.point_cloud.coord[nn_indices, 2]
        r, phi, theta = xyz2r_phi_theta(x, y, z)
        rbfi = Rbf(r, phi, theta, value, function='cubic')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        ri, phii, thetai = xyz2r_phi_theta(xi, yi, zi)
        intp_val = rbfi(ri, phii, thetai)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class RbfInterpolatorCartesian:
    def __init__(self, point_cloud):
        print('RbfInterpolatorCartesian')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        x, y, z = self.point_cloud.coord[nn_indices, 0], self.point_cloud.coord[nn_indices, 1], \
                  self.point_cloud.coord[nn_indices, 2]
        rbfi = Rbf(x, y, z, value, function='cubic')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        intp_val = rbfi(xi, yi, zi)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class RbfInterpolatorIDW:
    def __init__(self, point_cloud):
        print('RbfInterpolatorIDW')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        x, y, z = self.point_cloud.coord[nn_indices, 0], self.point_cloud.coord[nn_indices, 1], \
                  self.point_cloud.coord[nn_indices, 2]
        rbfi = Rbf(x, y, z, value, function='inverse')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        intp_val = rbfi(xi, yi, zi)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class RbfInterpolatorGaussian:
    def __init__(self, point_cloud):
        print('RbfInterpolatorGaussian')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        x, y, z = self.point_cloud.coord[nn_indices, 0], self.point_cloud.coord[nn_indices, 1], \
                  self.point_cloud.coord[nn_indices, 2]
        rbfi = Rbf(x, y, z, value, function='gaussian')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        intp_val = rbfi(xi, yi, zi)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class NearestNBInterpolator:
    def __init__(self, point_cloud):
        print('NearestNBInterpolator')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        coord = self.point_cloud.coord[nn_indices]
        rbfi = NearestNDInterpolator(coord, value)

        local_grid = local_grid.reshape([-1, 3])
        intp_val = rbfi(local_grid)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class LinearNBInterpolator:
    def __init__(self, point_cloud):
        print('LinearNBInterpolator')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):
        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        coord = self.point_cloud.coord[nn_indices]
        rbfi = LinearNDInterpolator(coord, value)

        local_grid = local_grid.reshape([-1, 3])
        intp_val = rbfi(local_grid)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class IDWNBInterpolator:
    def __init__(self, point_cloud):
        print('IDWInterpolator')
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):
        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]

        value = u[nn_indices]
        coord = self.point_cloud.coord[nn_indices]
        rbfi = IDWInterpolator(coord, value)

        local_grid = local_grid.reshape([-1, 3])
        intp_val = rbfi(local_grid)

        intp_val = intp_val.reshape(grid_shape)

        NaNValueCheck(intp_val)
        # self.plot(x, y, z, xi, yi, zi, value, intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return


class NaNValueCheck:
    def __init__(self, intp_val):
        number_of_nan_value = np.isnan(intp_val).sum()
        total_number_of_value = np.size(intp_val)

        percentage_of_nan_value = number_of_nan_value / total_number_of_value

        if np.isnan(intp_val).any() is True:
            raise ValueError('There exist an NaN value in interpolated_value')
        return
