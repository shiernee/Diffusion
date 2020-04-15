import sys
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\Diffusion\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\FileIO\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Utils\src\\utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np
from Utils import Utils

# analytic solution
def analytic_sln(t, theta):
    return np.exp(-2 * t.reshape([-1, 1])) * np.cos(theta.reshape([1, -1]))


def plot_analytic_sln(coord, u_exact, dt):
    step = int(u_exact.shape[1] / 8)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(331, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, 0], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(0 * dt))
    ax2 = fig2.add_subplot(332, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * dt))
    ax2 = fig2.add_subplot(333, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 2], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 2 * dt))
    ax2 = fig2.add_subplot(334, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 3], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 3 * dt))
    ax2 = fig2.add_subplot(335, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 4], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 4 * dt))
    ax2 = fig2.add_subplot(336, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 5], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 5 * dt))
    ax2 = fig2.add_subplot(337, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 6], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 6 * dt))
    ax2 = fig2.add_subplot(338, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 7], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 7 * dt))
    ax2 = fig2.add_subplot(339, projection='3d')
    cbar = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_exact[:, step * 8], vmin=-1, vmax=1)
    ax2.title.set_text('t:{:.2f}'.format(step * 8 * dt))
    plt.tight_layout()
    return

if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case2_sphere/forward/'
    START_TIME = 0
    END_TIME = 0.0005

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    i = 3
    physics_model_instances = fileio.read_physics_model_instance(i, model='diffusion_liang')
    point_cloud_instances = fileio.read_point_cloud_instance(i)

    coord = point_cloud_instances['coord']
    t = physics_model_instances['t']
    u = physics_model_instances['u']
    D = physics_model_instances['D']
    c = physics_model_instances['c']

    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    vr = ViewResultsUtils()
    vr.assign_x_y_z(x, y, z)
    no_pt = x.shape[0]
    vr.assign_no_pt(no_pt)
    vr.assign_u_t(u, t)

    vr.plot_theta_phi_V(START_TIME, END_TIME, u, no_of_plot=9)
    fileio.save_spatialV_png_file(i)

    vr.plot_u_theta(START_TIME, END_TIME, u, skip_dt=0.01)
    fileio.save_png_file(i, model='diffusion', fig_name='u_theta')

    # ===== analytical solution ====
    ut = Utils()
    r, phi, theta = ut.xyz2sph(coord)
    u_exact = analytic_sln(t, theta)
    # plot_analytic_sln(coord, u_exact)
    vr.plot_theta_phi_V(START_TIME, END_TIME, u_exact, no_of_plot=9)
    fileio.save_png_file(i, model='diffusion_analytic', fig_name='u_theta')

    err = np.mean(abs(u-u_exact), axis=1)
    # vr.plot_u_1D(skip_dt=0.1)
    # vr.plot_u_2D(skip_dt=0.2)


