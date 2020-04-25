from src.pointcloud.GeneratePoints import GeneratePoints
from src.utils.InterpolatedSpacing import InterpolatedSpacing
import numpy as np
import copy as cp

class Parameters:
    def __init__(self):

        self.forward_folder = '../data/case2_sphere/forward/'
        # HK : this got to be any number, small and big
        self.no_pt = 12
        self.radius_limit = 10
        self.duration = 0.1
        self.order_acc = 2   # int - central difference by taking two elements right and left each to compute gradient
        self.D = 1.0
        self.neighbours = {'nn_algorithm': 'kd_tree_radius',
                           'n_neighbors': 30,
                           'radius': self.radius_limit}
        self.interpolate_method = {'interpolate_type': 'Rbf_sph',
                                   'function': 'cubic'}
        self.interpolated_spacing_method = {'is_method': 'min_dist',
                                            'interpolated_spacing': None}
        self.u0_method = 'cos_theta'

        self.u0 = None
        self.dt = None

    def set_u_initial(self, **kwargs):
        """

        :param kwargs: {r, phi, theta}
        :return:
        """
        method = self.u0_method
        if method == 'cos_theta':
            theta = kwargs.get('theta')
            self.u0 = np.cos(theta)
            print('set u_initial')
        else:
            print('u_initial is not set')
        return

    def compute_dt(self, interpolated_spacing):
        self.dt = 0.1 * interpolated_spacing ** 2
        return cp.copy(self.dt)

