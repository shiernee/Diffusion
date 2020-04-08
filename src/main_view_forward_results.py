import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case4_LAF_21_52_36/forward/'
    START_TIME = 0
    END_TIME = 500

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    i = 1
    physics_model_instances = fileio.read_physics_model_instance(i, model='diffusion')
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

    vr.plot_theta_phi_V(START_TIME, END_TIME, no_of_plot=9)
    fileio.save_spatialV_png_file(i)

    vr.plot_u_theta(START_TIME, END_TIME, skip_dt=0.01)
    fileio.save_png_file(i, model='diffusion', fig_name='u_theta')

    # vr.plot_u_1D(skip_dt=0.1)
    # vr.plot_u_2D(skip_dt=0.2)


