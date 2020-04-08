import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')

import numpy as np
import pandas as pd
from FileIO import FileIO
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import Utils
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def instance_to_dict(coord, t, u, D, c):
    instance = \
        {'coord': coord,
         't': t,
         'u': u,
         'D': D,
         'c': c,
         }

    return instance

def view_3D_V(coord, V):
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cbar = ax.scatter(x, y, z, s=100, c=np.squeeze(V), marker='.')
    fig.colorbar(cbar)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return



if __name__ == '__main__':
    # === READ PREDICTED PARAMETERS FROM SAV  =====
    forward_folder = '../data/case4_LAF_21_52_36/forward/'

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    i = 1

    # === read from patients csv ====================
    dataframe = pd.read_csv('{}/coordinates.csv'.format(forward_folder))
    xtra_dataframe = pd.read_csv('{}/coordinates_to_be_deleted.txt'.format(forward_folder), sep=' ', header=None)
    coord_raw = dataframe[['ptx', 'pty', 'ptz']].values
    coord_to_be_deleted = xtra_dataframe.values

    # ====== remove extra coordinates ======
    idx=np.ones(len(coord_raw))
    for value in coord_to_be_deleted:
        aa = np.sum(abs(value - coord_raw), axis=1)
        idx[np.where(aa < 1e-10)[0]] = 0
    idx = np.array(idx).squeeze()
    coord = coord_raw[idx==1]

    no_pt = len(coord)
    u = np.zeros([no_pt, 1])
    t = np.zeros([no_pt, 1])
    D = np.ones([no_pt, 1])
    c = np.ones([no_pt, 1]) * 0

    local_axis1 = None
    local_axis2 = None

    view_3D_V(coord, u)

    ut = Utils()
    r, phi, theta = ut.xyz2sph(coord)
    print('n_pt: {}, ori_dx: {}'.format(len(coord), np.sqrt(np.pi * r.mean()**2 / len(coord))))

    # ============= interpolate points ========
    # n_neighbours = 8
    # nbrs = NearestNeighbors(n_neighbours, algorithm='kd_tree').fit(coord)
    # dist, nn_indices = nbrs.kneighbors(coord)
    # nn_coord = coord[nn_indices]
    # # local_axis1, local_axis2 = compute_local_axis(nn_coord, coord)
    # vec1 = nn_coord[:, int(n_neighbours/2) - 1] - coord
    # vec2 = nn_coord[:, int(n_neighbours/2)] - coord
    #
    # mag1, mag2 = np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1)
    # unit_vec1 = vec1 / np.expand_dims(mag1, axis=1)
    # unit_vec2 = vec2 / np.expand_dims(mag2, axis=1)
    #
    # median_pt2pt_dist = np.median(np.array([mag1, mag2]))
    # max_pt2pt_dist = np.array([mag1, mag2]).max()
    # min_pt2pt_dist = np.array([mag1, mag2]).min()
    #
    # thres = 5
    # idx1 = np.argwhere(mag1 > thres).squeeze()
    # idx2 = np.argwhere(mag2 > thres).squeeze()
    #
    # scale = int(median_pt2pt_dist)
    # intp_coord1 = coord[idx1] + unit_vec1[idx1] * scale
    # intp_coord2 = coord[idx1] - unit_vec1[idx1] * scale
    # intp_coord3 = coord[idx2] + unit_vec2[idx2] * scale
    # intp_coord4 = coord[idx2] - unit_vec2[idx2] * scale
    #
    # assert np.ndim(intp_coord1) == 2, 'intp_coord1 must be 2D array'
    # assert np.ndim(intp_coord2) == 2, 'intp_coord1 must be 2D array'
    # assert np.ndim(intp_coord3) == 2, 'intp_coord1 must be 2D array'
    # assert np.ndim(intp_coord4) == 2, 'intp_coord1 must be 2D array'
    #
    # coord_incl_intp_coord = np.array([])
    # coord_incl_intp_coord = np.concatenate((intp_coord1, intp_coord2), axis=0)
    # coord_incl_intp_coord = np.concatenate((coord_incl_intp_coord, intp_coord3), axis=0)
    # coord_incl_intp_coord = np.concatenate((coord_incl_intp_coord, intp_coord4), axis=0)
    # coord_incl_intp_coord = np.concatenate((coord_incl_intp_coord, coord), axis=0)
    # coord = coord_incl_intp_coord.copy()
    #
    # print(coord.shape)
    #
    # no_pt = len(coord)
    # V = np.zeros([no_pt, 1])
    # V = np.zeros([no_pt, 1])
    # v = np.zeros([no_pt, 1])
    # D = np.ones([no_pt, 1])
    # c = np.ones([no_pt, 1])
    #
    # a = np.ones([no_pt, 1])
    # epsilon = np.ones([no_pt, 1])
    # beta = np.ones([no_pt, 1])
    # gamma = np.ones([no_pt, 1])
    # delta = np.ones([no_pt, 1])
    # applied_current = np.ones([no_pt, 1])
    #
    # local_axis1 = None
    # local_axis2 = None
    #
    # view_3D_V(coord_incl_intp_coord, V)

    ## ================ assign initial boundary condition ============
    idx_to_have_activation = random.sample(range(0, len(coord)), 2)
    u[idx_to_have_activation] = 100.0
    region_to_apply_current = coord[idx_to_have_activation]

    view_3D_V(coord, u)
    # ================================================================

    instance = instance_to_dict(coord, t, u, D, c)

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.write_generated_instance(instance)


'''
    mesh = om.TriMesh()
    vh = []
    for value in coord:
        vh.append(mesh.add_vertex(value))
    fh = mesh.add_face(vh)

    view_phi_theta_V(coord, V)
    view_3D_V(coord_incl_intp_coord, V)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    o3d.visualization.draw_geometries([pcd])
    tetra_mesh, pt_map = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1, None)


    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    # Radius oulier removal
    cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=5)
    display_inlier_outlier(pcd, ind)


    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    '''


