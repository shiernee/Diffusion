
from scipy.interpolate import Rbf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RbfInterpolator:

    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        return

    # =================================================
    def eval(self, grid, u):

        grid_shape = grid[0].shape[:2]

        local_grid = grid[0]
        nn_indices = grid[1]
        x, y, z = self.point_cloud.coord[nn_indices, 0], self.point_cloud.coord[nn_indices, 1], \
                  self.point_cloud.coord[nn_indices, 2]
        value = u[nn_indices]
        rbfi_s = Rbf(x, y, z, value, function='cubic')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]

        intp_val = rbfi_s(xi, yi, zi)
        intp_val = intp_val.reshape(grid_shape)  # HK

        # self.plot(x, y, z, xi, yi, zi, value, self.intp_val)

        return intp_val  # see variables.py lines 50-70
    # =================================================

    def plot(self, x, y, z, xi, yi, zi, value, valuei):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=value, vmin=value.min(), vmax=value.max())
        ax.scatter(xi, yi, zi, c=valuei, vmin=value.min(), vmax=value.max())
        plt.waitforbuttonpress()
        return

