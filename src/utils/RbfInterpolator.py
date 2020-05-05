
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

        value = u[nn_indices]
        x, y, z = self.point_cloud.coord[nn_indices, 0], self.point_cloud.coord[nn_indices, 1], \
                  self.point_cloud.coord[nn_indices, 2]
        rbfi = Rbf(x, y, z, value, function='cubic')

        local_grid = local_grid.reshape([-1, 3])
        xi, yi, zi = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]

        # intp_val = rbfi(xi, yi, zi)
        # intp_val = intp_val.reshape(grid_shape)

        r, phi, theta = self.xyz2sph(self.point_cloud.coord[nn_indices])
        rbfi_s = Rbf(r, phi, theta, value, function='cubic')
        ri, phii, thetai = self.xyz2sph(local_grid)
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

