from src.utils.InterpolatedSpacing import InterpolatedSpacing

import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
from Utils.src.utils.Utils import Utils
import copy as cp

class PointCloud:
    # ===== this model will compute all the neccesary parameter needed for forward solver====== #
    def __init__(self,cart_coord,
                 interpolated_spacing_method,
                 neighbors,
                 order_acc):

        self.coord = None
        self.no_pt = None

        self.nbrs = None
        self.dist_nn = None
        self.nn_indices = None

        self.local_axis1 = None  # return from interpolated_parameter
        self.local_axis2 = None

        self.interpolated_spacing = None
        self.local_grid = None

        self.assign_coord_(cart_coord)
        self.compute_interpolated_spacing_(interpolated_spacing_method)
        self.compute_nn_indices_neighbor_(neighbors)
        self.compute_local_axis_()
        self.make_local_grids_(order_acc)

    def assign_coord_(self, coord):
        assert np.ndim(coord) == 2, 'coordinates dimension should be 2D array'
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'

        self.coord = coord.copy()
        self.no_pt = int(np.shape(self.coord)[0])
        return

    def compute_interpolated_spacing_(self, method):
        intp_spc_method= method.get('is_method')
        interpolated_spacing = method.get('interpolated_spacing')
        is_ = InterpolatedSpacing(intp_spc_method, self.coord, interpolated_spacing)
        self.interpolated_spacing = cp.copy(is_.interpolated_spacing)
        return

    def compute_nn_indices_neighbor_(self, neighbours):
        """

        :param neighbours: dict {nn_algorithm, n_neighbors, radius}
        :return:
        """
        algorithm = neighbours.get('nn_algorithm')
        print('Neighbour points was found using {}'.format(algorithm))
        # === find the n_neighbours coordinates for each coordinate. ==== #
        if algorithm == 'kd_tree_no_neighbour':
            n_neighbors = neighbours.get('n_neighbors')
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(self.coord)
            self.dist, self.nn_indices = self.nbrs.kneighbors(self.coord)
            self.dist, self.nn_indices = list(self.dist), list(self.nn_indices)

        if algorithm == 'kd_tree_radius':
            radius = neighbours.get('radius')
            self.nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(self.coord)
            self.dist, self.nn_indices = self.nbrs.radius_neighbors(self.coord)

        if algorithm == 'ball_tree':
            n_neighbors = neighbours.get('n_neighbors')
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(self.coord)
            self.dist, self.nn_indices = self.nbrs.kneighbors(self.coord)

        return

    def compute_local_axis_(self):
        local_axis1, local_axis2 = [], []
        for i in range(self.no_pt):
            nn_indices = self.nn_indices[i].copy()
            dist = self.dist[i].copy()
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

    def make_local_grids_(self, order_acc, order_derivative=2):
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

