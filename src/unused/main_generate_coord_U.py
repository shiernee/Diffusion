import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import csv
from FileIO import FileIO
from sklearn.externals import joblib
from Utils import Utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def instance_to_dict(coord, t, u, D, c):
    instance = \
        {'coord': coord,
         't': t,
         'u': u,
         'D': D,
         'c': c,
         }

    return instance

def generate_point_regular_grid(dx, max_x, max_y):
    x = np.arange(0, max_x + dx, dx)
    y = np.arange(0, max_y + dx, dx)
    X, Y = np.meshgrid(x, y)
    coord = np.zeros([len(X.flatten()), 3])
    coord[:, 0] = X.flatten()
    coord[:, 1] = Y.flatten()

    return coord

def generate_points_2D(dx, max_x, max_y):
    no_pt = int(max_x / dx + 1) * int(max_x / dx + 1)
    x = np.random.rand(no_pt, ) * max_x
    y = np.random.rand(no_pt, ) * max_y
    coord = np.zeros([no_pt, 3])
    coord[:, 0] = x
    coord[:, 1] = y

    return coord

def add_noise_to_u(u):
    mean = 10.0
    stdev = 1.0
    noise = np.random.normal(loc=mean, scale=stdev, size=u.shape)
    u_added_noise = u + noise

    return u_added_noise

def generate_points_sphere(n_point, sph_radius):
    ut = Utils()
    random_no = np.random.uniform(0.0, 1.0, n_point)
    theta = np.arccos(2 * random_no - 1.0)
    random_no = np.random.uniform(0.0, 1.0, n_point)
    phi = 2 * np.pi * random_no

    sph_coord = np.zeros([n_point, 3])
    sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2] = sph_radius, phi, theta
    x, y, z = ut.sph2xyz(sph_coord)

    cart_coord = np.zeros([n_point, 3])
    cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2] = x, y, z

    return cart_coord


def view_3D_V(coord, V):
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cbar = ax.scatter(x, y, z, s=100, c=np.squeeze(V))
    fig.colorbar(cbar)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return

def view_phi_theta_V(coord, V):
    ut = Utils()

    _, phi, theta = ut.xyz2sph(coord)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cbar = ax.scatter(theta, phi, c=np.squeeze(V))
    fig.colorbar(cbar)
    ax.set_xlabel('theta')
    ax.set_ylabel('phi')
    plt.show()


if __name__ == '__main__':

    forward_folder = '../data/case3_sphere_D1_c0/forward/'
    # === read file ===
    u_file_csv = 'u_file_from_set_DC'
    DC_file_csv = 'set_DC'

    reader = csv.reader(open('{}/{}.csv'.format(forward_folder, u_file_csv)))
    data = np.array(list(reader)[1:], dtype='float64')
    coord, t, u = data[:, :3], data[:, 3], data[:, 4]

    reader = csv.reader(open('{}/{}.csv'.format(forward_folder, DC_file_csv)))
    data = np.array(list(reader)[1:], dtype='float64')
    coord2, D, c = data[:, :3], data[:, 3], data[:, 4]

    np.array_equal(coord, coord2), 'coord1 and coord2 are equal'

    # === sphere ===
    # n_points = 5000
    # coord = generate_points_sphere(n_point=n_points, sph_radius=1.0)
    #
    # # =========== COMPULSORY ================================
    # u = np.zeros([len(coord), 1]) * 25
    # t = np.zeros([len(coord), 1])
    # D = np.ones([len(coord), 1])
    # c = np.ones([len(coord), 1]) * 0
    #
    # # ================ assign initial boundary condition ============
    # # ===== sphere ================
    # ut = Utils()
    # r, phi, theta = ut.xyz2sph(coord)
    # north_pole_indx = np.where(abs(theta - 0) < 0.2)
    # south_pole_indx = np.where(abs(theta - np.pi) < 0.2)
    # equator_indx = np.where(abs(theta - np.pi / 2) < 0.1)
    # region_to_apply_current = coord[north_pole_indx, :]
    #
    # u[north_pole_indx] = 100
    # u[south_pole_indx] = 100
    # u[equator_indx] = 0
    #
    # view_phi_theta_V(coord, u)
    # view_3D_V(coord, u)
    # ================================================================

    # u_added_noise = add_noise_to_u(u)

    instance = instance_to_dict(coord, t, u, D, c)

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.write_generated_instance(instance)










