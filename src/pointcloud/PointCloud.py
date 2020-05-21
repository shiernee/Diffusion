from src.utils.InterpolatedSpacing import InterpolatedSpacing

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import time
import copy as cp
import pandas as pd

min_rad_to_search_nn = 0.2  # minimum radius to search nearest neighbour


class PointCloud:
    def __init__(self, coord, grid_length, local_grid_resolution):

        # ==========================================================
        self.coord = coord
        self.no_pt = len(self.coord)

        self.nbrs = None
        self.dist_nn = None
        self.nn_indices = None

        self.local_axis1 = None
        self.local_axis2 = None

        self.local_grid_resolution = local_grid_resolution
        self.interpolated_spacing = grid_length / (self.local_grid_resolution - 1)
        self.local_grid = None

        # ============================================================
        # self._compute_interpolated_spacing(min_intp_spacing)
        # self._compute_nn_indices_neighbor('kd_tree_no_neighbour', n_neighbors=60)
        self._compute_nn_indices_neighbor('kd_tree_radius', nn_radius_limit=grid_length)
        self._compute_local_axis()
        self._make_local_grids()

    # ===========================================================================
    def _compute_interpolated_spacing(self, min_intp_spacing):
        spacing = distance.pdist(self.coord).min()
        if spacing < min_intp_spacing:
            self.interpolated_spacing = min_intp_spacing
        else:
            self.interpolated_spacing = spacing
        return

    # ===========================================================================
    def _compute_nn_indices_neighbor(self, nn_algorithm, nn_radius_limit=None, n_neighbors=None):
        """
        :param neighbours: dict {nn_algorithm, n_neighbors, radius}
        :return:
        """
        if nn_radius_limit < min_rad_to_search_nn:  # minimum radius to search nearest neighbour is 0.2
            nn_radius_limit = min_rad_to_search_nn

        algorithm = nn_algorithm
        # === find the n_neighbours coordinates for each coordinate. ==== #
        if algorithm == 'kd_tree_no_neighbour':
            if n_neighbors is None:
                raise ValueError('n_neighbors is None, NearestNeighbour cannot performed')
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(self.coord)
            self.dist_nn, self.nn_indices = self.nbrs.kneighbors(self.coord)
            self.dist_nn, self.nn_indices = list(self.dist_nn), list(self.nn_indices)

        if algorithm == 'kd_tree_radius':
            if nn_radius_limit is None:
                raise ValueError('nn_radius_limit is None, NearestNeighbour within radius cannot performed')
            self.nbrs = NearestNeighbors(radius=nn_radius_limit, algorithm='kd_tree').fit(self.coord)
            self.dist_nn, self.nn_indices = self.nbrs.radius_neighbors(self.coord)
        return

    # ============================================================
    def _compute_local_axis(self):
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

    # ============================================================
    def _make_local_grids(self):

        mididx = int(self.local_grid_resolution/2)
        self.local_grid = np.zeros([self.no_pt, self.local_grid_resolution, self.local_grid_resolution, 3])

        for i in range(self.no_pt):
            for row in range(self.local_grid_resolution):
                for col in range(self.local_grid_resolution):
                    self.local_grid[i, row, col] = self.coord[i] + \
                                              (col - mididx) * self.interpolated_spacing * self.local_axis2[i] + \
                                              (row - mididx) * self.interpolated_spacing * self.local_axis1[i]

        return

    def get_grid_list(self):
        return list(zip(self.local_grid.copy(), self.nn_indices.copy(), self.dist_nn.copy()))

    # ============================================================
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

    # ============================================================
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

