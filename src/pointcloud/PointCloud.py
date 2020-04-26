from src.utils.InterpolatedSpacing import InterpolatedSpacing

import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import copy as cp
import pandas as pd


class PointCloud:
    # ===== this model will compute all the neccesary parameter needed for forward solver====== #
    def __init__(self, case):

        # ==========================================================
        self.coord = None
        self.no_pt = None

        self.nbrs = None
        self.dist_nn = None
        self.nn_indices = None

        self.local_axis1 = None  # return from interpolated_parameter
        self.local_axis2 = None

        self.interpolated_spacing = None
        self.local_grid = None

        # ============================================================

        coordinate_file = '../data/{}/coordinates.csv'.format(case)
        self.parameter_file = '../data/{}/param_template.csv'.format(case)

        self.coord = pd.read_csv(coordinate_file).values
        self.no_pt = len(self.coord)

        self.param = pd.read_csv(self.parameter_file)
        nn_algorithm = self.param['nn_algorithm'].values[0]
        nn_radius_limit = self.param['nn_radius_limit'].values[0]
        interpolated_spacing_method = self.param['interpolated_spacing_method'].values[0]
        interpolated_spacing_value = self.param['interpolated_spacing_value'].values[0]

        # ===========================================================================

        self.compute_interpolated_spacing_(interpolated_spacing_method, interpolated_spacing_value)
        self.compute_nn_indices_neighbor_(nn_algorithm=nn_algorithm, nn_radius_limit=nn_radius_limit)
        self.compute_local_axis_()
        self.make_local_grids_()

    def compute_interpolated_spacing_(self, interpolated_spacing_method, interpolated_spacing_value):
        is_ = InterpolatedSpacing(interpolated_spacing_method, self.coord, interpolated_spacing_value)
        self.interpolated_spacing = cp.copy(is_.interpolated_spacing)
        self.write_interpolated_spacing_to_parameter_file()
        return

    def write_interpolated_spacing_to_parameter_file(self):
        self.param['interpolated_spacing_value'] = cp.copy(self.interpolated_spacing)
        self.param.to_csv(self.parameter_file, index=False, index_label=False)
        return


    def compute_nn_indices_neighbor_(self, nn_algorithm, nn_radius_limit=None, n_neighbors=None):
        """

        :param neighbours: dict {nn_algorithm, n_neighbors, radius}
        :return:
        """
        algorithm = nn_algorithm
        print('Neighbour points was found using {}'.format(algorithm))
        # === find the n_neighbours coordinates for each coordinate. ==== #
        if algorithm == 'kd_tree_no_neighbour':
            if n_neighbors is None:
                raise ValueError('n_neighbors is None, NearestNeighbour cannot performed')
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(self.coord)
            self.dist_nn, self.nn_indices = self.nbrs.kneighbors(self.coord)
            self.dist_nn, self.nn_indices = list(self.dist), list(self.nn_indices)

        if algorithm == 'kd_tree_radius':
            if nn_radius_limit is None:
                raise ValueError('nn_radius_limit is None, NearestNeighbour within radius cannot performed')
            self.nbrs = NearestNeighbors(radius=nn_radius_limit, algorithm='kd_tree').fit(self.coord)
            self.dist_nn, self.nn_indices = self.nbrs.radius_neighbors(self.coord)

        if algorithm == 'ball_tree':
            if n_neighbors is None:
                raise ValueError('n_neighbors is None, NearestNeighbour cannot performed')
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(self.coord)
            self.dist_nn, self.nn_indices = self.nbrs.kneighbors(self.coord)

        return

    def compute_local_axis_(self):
        local_axis1, local_axis2 = [], []
        for i in range(self.no_pt):
            nn_indices = self.nn_indices[i].copy()
            dist = self.dist_nn[i].copy()
            coord = self.coord[i].copy().reshape([1, -1])
            assert np.ndim(coord) == 2, 'nn_coord should be (1, 3 axes)'

            sorted_nn_indices = nn_indices[np.argsort(dist)]
            idx_A, idx_B = sorted_nn_indices[1], sorted_nn_indices[2]

            # compute local axis
            v_a = self.coord[idx_A, :] - coord  # vector A
            v_b = self.coord[idx_B, :] - coord  # vector B

            v_n = np.cross(v_a, v_b)  ## normal vector to plane AB and interest pt
            v_n = (v_n / np.linalg.norm(v_n))  ##unit vector normal
            v_a_d2 = np.cross(v_n, v_a)

            local_axis1_tmp = v_a / np.linalg.norm(v_a, axis=1, keepdims=True)
            local_axis2_tmp = v_a_d2 / np.linalg.norm(v_a_d2, axis=1, keepdims=True)

            local_axis1.append(local_axis1_tmp.copy())
            local_axis2.append(local_axis2_tmp.copy())

        self.local_axis1 = np.array(local_axis1, dtype='float64')
        self.local_axis2 = np.array(local_axis2, dtype='float64')

        return

    def make_local_grids_(self, order_acc=2, order_derivative=2):
        fd_coeff_length = self.fd_coeff_length(order_acc)
        local_interp_size = self.compute_no_pt_needed_for_interpolation(fd_coeff_length, order_derivative)
        local_grid = np.zeros([self.no_pt, local_interp_size, local_interp_size, 3])

        for i in range(self.no_pt):
            for row in range(-int(local_interp_size/2), int(local_interp_size/2)+1):
                for col in range(-int(local_interp_size/2), int(local_interp_size/2)+1):
                    local_grid[i, row, col] = self.coord[i] + \
                                               col * self.interpolated_spacing * self.local_axis2[i] + \
                                               row * self.interpolated_spacing * self.local_axis1[i]
        self.local_grid = local_grid
        return

    @staticmethod
    def fd_coeff_length(order_acc):
        return order_acc + 1

    @staticmethod
    def compute_no_pt_needed_for_interpolation(fd_coeff_length, order_derivative):
        return fd_coeff_length * order_derivative - 1

    def grid_list(self):
        return list(zip(self.local_grid.copy(), self.nn_indices.copy()))

    @staticmethod
    def sph2xyz(sph_coord):
        sph_radius, phi, theta = sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2]
        x = sph_radius * np.sin(theta) * np.cos(phi)
        y = sph_radius * np.sin(theta) * np.sin(phi)
        z = sph_radius * np.cos(theta)
        return x, y, z

    def instance_to_dict(self):
        physics_model_instance = \
            {'coord': self.coord,
             'no_pt': self.no_pt,
             'nbrs': self.nbrs,
             'dist_nn': self.dist_nn,
             'nn_indices': self.nn_indices,
             'nn_coord': self.nn_coord,
             'local_axis1': self.local_axis1,
             'local_axis2': self.local_axis2,
              }

        return physics_model_instance

    def assign_read_point_cloud_instances(self, point_cloud_instances):
        self.coord = point_cloud_instances['coord']
        self.no_pt = point_cloud_instances['no_pt']
        self.nbrs = point_cloud_instances['nbrs']
        self.dist_nn = point_cloud_instances['dist_nn']
        self.nn_indices = point_cloud_instances['nn_indices']
        self.nn_coord = point_cloud_instances['nn_coord']
        self.local_axis1 = point_cloud_instances['local_axis1']
        self.local_axis2 = point_cloud_instances['local_axis2']
        print('Finish assigning read instances to Point_Cloud instances')
        return

