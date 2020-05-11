from os import sys, path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from scipy.interpolate import Rbf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils.Utils import xyz2r_phi_theta


class RbfInterpolatorSph:

    def __init__(self, point_cloud):
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
        rbfi_s = Rbf(r, phi, theta, value, function='cubic')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        ri, phii, thetai = xyz2r_phi_theta(xi, yi, zi)
        intp_val = rbfi_s(ri, phii, thetai)
        intp_val = intp_val.reshape(grid_shape)

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
        # import matplotlib.pyplot as plt
        # plt.scatter(grid[0].reshape([-1, 3])[:5, 0], grid[0].reshape([-1, 3])[:5, 0])
        # plt.scatter(grid[0].reshape([-1, 3])[12, 0], grid[0].reshape([-1, 3])[12, 0])
        # plt.scatter(x, y)
        # plt.scatter(grid[0].reshape([-1, 3])[:, 0], grid[0].reshape([-1, 3])[:, 0], c=intp_val.reshape([-1, ]))
        # plt.scatter(grid[1].reshape([-1, 3])[:, 0], grid[1].reshape([-1, 3])[:, 0], c=value.reshape([-1, ]))
        # plt.show()
        # quit()

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


