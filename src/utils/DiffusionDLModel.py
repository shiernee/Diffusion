import tensorflow as tf
import numpy as np
from Utils import Utils


class DiffusionDLModel(tf.keras.Model):
    def __init__(self):
        super(DiffusionDLModel, self).__init__()

        self.nn_indices = None
        self.interpolated_spacing = None
        self.dist_intp_coord_axis1 = None
        self.dist_intp_coord_axis2 = None
        self.t = None
        self.u = None
        self.no_pt = None

        self.tf_weight_D = None
        self.tf_weight_c = None
        self.tf_lowest_weight_D = None
        self.tf_lowest_weight_c = None

        self.nn_u = None
        self.nn_weight_D = None

        self.tf_intp_u_axis1 = None
        self.tf_intp_u_axis2 = None
        self.tf_dudt = None

        self.ut = Utils()

        self.training_loss = None
        self.n_training_epoch = None

    def assign_nn_indices(self, nn_indices):
        """

        :param nn_indices: array (pts, nn)
        :return: void
        """
        assert np.ndim(nn_indices) == 2, 'nn_indices shape have 2 dimension (pts, no_of_neighbour_pts'
        self.nn_indices = nn_indices
        return

    def assign_interpolated_spacing(self, interpolated_spacing):
        """

        :param interpolated_spacing: float
        :return: void
        """
        assert isinstance(interpolated_spacing, float), 'interpolated_spacing must be a float'
        self.interpolated_spacing = interpolated_spacing
        return

    def assign_dist_intp_coord_axis1(self, dist_intp_coord_axis1):
        """

        :param dist_intp_coord_axis1: 3D array (eg. 5000, 9, 8)
        :return: void
        """
        assert np.ndim(dist_intp_coord_axis1) == 3, 'dist_intp_coord_axis1 must be 3D (pts, intp_pts, nn_pts'
        self.dist_intp_coord_axis1 = dist_intp_coord_axis1
        return

    def assign_dist_intp_coord_axis2(self, dist_intp_coord_axis2):
        """

        :param dist_intp_coord_axis2: 3D array (eg. 5000, 9, 8)
        :return: void
        """
        assert np.ndim(dist_intp_coord_axis2) == 3, 'dist_intp_coord_axis1 must be 3D (pts, intp_pts, nn_pts'

        self.dist_intp_coord_axis2 = dist_intp_coord_axis2
        return

    def assign_t(self, t):
        """

        :param t: 1D array (timept, )
        :return: void
        """
        assert np.ndim(t) == 1, 't must be 1D array (timept, )'
        self.t = t
        return

    def assign_u(self, u):
        """

        :param u: 2D array (timept, pts)
        :return: void
        """
        assert np.ndim(u) == 2, 'u must be 2D array (timept, pts)'
        self.u = u
        return

    def compute_no_pt(self):
        self.no_pt = self.u.shape[-1]
        return

    def assign_tf_lowest_weight_D(self, tf_lowest_weight_D):
        self.tf_lowest_weight_D = tf_lowest_weight_D.__copy__()
        return

    def assign_tf_lowest_weight_c(self, tf_lowest_weight_c):
        self.tf_lowest_weight_c = tf_lowest_weight_c.__copy__()
        return

    # def compute_nn_u(self):
    #     nn_indices = self.nn_indices.copy()
    #     u = self.u.copy()
    #
    #     assert np.ndim(u[:, nn_indices]) == 3, 'nn_u shape must have 3 dimension'
    #     self.nn_u = u[:, nn_indices]
    #     return

    def interpolate_u_axis1_axis2(self, order_acc):
        """

        :param order_acc: int
        :return: array, array
        """
        assert isinstance(order_acc, int), 'order_acc must be an integer'

        n_neighbor = self.nn_indices.shape[-1]
        dist_intp_coord_axis1 = self.dist_intp_coord_axis1.reshape([-1, n_neighbor])
        dist_intp_coord_axis2 = self.dist_intp_coord_axis2.reshape([-1, n_neighbor])

        intp_u_axis1 = self.interpolate_u(dist_intp_coord_axis1, order_acc)
        intp_u_axis2 = self.interpolate_u(dist_intp_coord_axis2, order_acc)
        return intp_u_axis1, intp_u_axis2

    def interpolate_u(self, dist_intp_coord_axis, order_acc):
        """

        :param dist_intp_coord_axis: 3D array (timept, intp_coord, nn)
        :param order_acc: int
        :return: array
        """
        assert np.ndim(dist_intp_coord_axis) == 2, 'dist_intp_coord_axis must be 2D array ((no_pts, nn)'
        assert isinstance(order_acc, int), 'order_acc must be an integer'

        u = self.u.copy()
        nn_u = u[:, self.nn_indices]

        duration = self.t[-1]
        dt = self.t[2] - self.t[1]
        time_pt = np.linspace(0, duration, int(duration / dt) + 1, endpoint=True, dtype='float64')

        # assert nn_coord is not None, 'nn_coord is None'
        assert nn_u is not None, 'nn_D is None'

        # mid_point_ind = int(np.floor(dist_intp_coord_axis.shape[1]/2))
        n_neighbor = self.nn_indices.shape[-1]

        intp_u_axis = []
        for t in range(len(time_pt) - 1):
            nn_u_at_time_t = nn_u[t]
            u_at_time_t = u[t]
            intp_u_axis_tmp = self.ut.idw_interpolate(dist_intp_coord_axis, nn_u_at_time_t)
            # intp_u_axis_tmp[:, mid_point_ind] = u_at_time_t
            intp_u_axis.append(intp_u_axis_tmp)

        assert np.ndim(intp_u_axis) == 3, 'intp_u_axis must be 3D array (no_pt, no_coord, 5nn)'

        return intp_u_axis

    def assign_intp_u_axis1(self, intp_u_axis1):
        """

        :param intp_u_axis1: array
        :return:
        """
        self.tf_intp_u_axis1 = tf.convert_to_tensor(intp_u_axis1, dtype=tf.float64)
        return

    def assign_intp_u_axis2(self, intp_u_axis2):
        self.tf_intp_u_axis2 = tf.convert_to_tensor(intp_u_axis2, dtype=tf.float64)
        return

    def initialize_weight(self):
        self.tf_weight_D = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=1.0, stddev=0.1,
                                       dtype=tf.float64), trainable=True, name='D')
        # self.tf_weight_D.assign(np.ones(self.tf_weight_D.shape))
        self.tf_weight_c = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=0.0, stddev=0.1,
                                       dtype=tf.float64), trainable=False, name='c')
        self.tf_weight_c.assign(np.zeros(self.tf_weight_c.shape))
        return

    def compute_dudt(self):
        dudt = []
        u = self.u.copy()
        dt = self.t[1] - self.t[0]
        for n in range(len(u) - 1):
            tmp = np.array(u[n + 1], dtype='float64') - np.array(u[n], dtype='float64')
            dudt.append(tmp / dt)

        dudt = np.array(dudt, dtype='float64')
        dudt = np.expand_dims(dudt, -1)  #shape (time_pt, no_pt, 1)
        # dudt = np.expand_dims(dudt, -1)  #shape (time_pt, no_pt, 1, 1)
        return dudt

    def assign_dudt(self, dudt_true):
        self.tf_dudt = tf.convert_to_tensor(dudt_true, dtype=tf.float64)
        return

    def call_dl_model(self, tf_intp_u_axis1, tf_intp_u_axis2, tf_coeff_matrix_first_der, tf_coeff_matrix_second_der,
                      order_acc):

        tf.assert_equal(tf.rank(tf_intp_u_axis1), 3,
                        'tf_intp_u_axis1 should have 3 dimension (time_pt, n_coord, 5nn_u)')
        tf.assert_equal(tf.rank(tf_intp_u_axis2), 3,
                        'tf_intp_u_axis2 should have 3 dimension (time_pt, n_coord, 5nn_u)')
        tf.assert_equal(tf.rank(tf_coeff_matrix_first_der), 3,
                        'tf_coeff_matrix_first_der should have 3 dimension (1, 5, 3)')
        tf.assert_equal(tf.rank(tf_coeff_matrix_second_der), 3,
                        'tf_coeff_matrix_second_der should have 3 dimension (1, 3, 1)')

        tf_intp_D_axis1, tf_intp_D_axis2 = self.interpolate_D_axis(self.tf_weight_D, order_acc)
        tf_weight_c = self.tf_weight_c
        interpolated_spacing = self.interpolated_spacing

        tf_dudt_pred = self.tf_del_D_delV(tf_intp_u_axis1, tf_intp_u_axis2, tf_intp_D_axis1, tf_intp_D_axis2,
                                         tf_weight_c, interpolated_spacing,
                                         tf_coeff_matrix_first_der, tf_coeff_matrix_second_der)

        return tf_dudt_pred

    def interpolate_D_axis(self, tf_weight_D, order_acc):
        assert tf.is_tensor(tf_weight_D) is True, 'D input must be a tensor'
        tf.assert_equal(tf.rank(tf_weight_D), 3, 'tf_weight_D should have 3 dimension (1, n_coord, 1)')

        assert isinstance(order_acc, int), 'order_acc should be integer'

        D = tf.squeeze(tf_weight_D)  # shape(no_pt)
        nn_D = tf.gather(D, self.nn_indices)  # shape(no_pt, 8nn)

        # ======= intp_D_axis1 and intp_D_axis2 shape expected to be (n_coord, 5) ======
        tf_intp_D_axis1 = self.ut.tf_idw_interpolate(self.dist_intp_coord_axis1, nn_D)  # shape(n_coord, 5)
        tf_intp_D_axis2 = self.ut.tf_idw_interpolate(self.dist_intp_coord_axis2, nn_D)  # shape(n_coord, 5)

        intp_u_axis1_tmp = self.tf_intp_u_axis1.numpy()
        fd_coeff_length = order_acc + 1
        ind_start_position = int(np.floor(np.shape(intp_u_axis1_tmp)[-1] / 2) - np.floor(fd_coeff_length / 2))
        begin = [0, ind_start_position]
        size = [tf_intp_D_axis1.shape[0], fd_coeff_length]

        # shape(n_coord, 3)
        tf_intp_D_axis1 = tf.slice(tf_intp_D_axis1, begin, size)
        tf_intp_D_axis2 = tf.slice(tf_intp_D_axis2, begin, size)

        tf_intp_D_axis1 = tf.expand_dims(tf_intp_D_axis1, 0)  # shape (1, n_coord, 3)
        tf_intp_D_axis2 = tf.expand_dims(tf_intp_D_axis2, 0)  # shape (1, n_coord, 3)

        return tf_intp_D_axis1, tf_intp_D_axis2

    @staticmethod
    def tf_del_D_delV(tf_intp_u_axis1, tf_intp_u_axis2, tf_intp_D_axis1, tf_intp_D_axis2, tf_weight_c,
                   interpolated_spacing, tf_coeff_matrix_first_der, tf_coeff_matrix_second_der):

        tf.assert_equal(tf.rank(tf_intp_u_axis1), 3, 'tf_intp_u_axis1 shape must be (time_pt, no_pt, 5nn_u)')
        tf.assert_equal(tf.rank(tf_intp_u_axis2), 3, 'tf_intp_u_axis2 shape must be (time_pt, no_pt, 5nn_u)')
        tf.assert_equal(tf.rank(tf_intp_D_axis1), 3, 'tf_intp_D_axis1 shape must be (1, no_pt, 3)')
        tf.assert_equal(tf.rank(tf_intp_D_axis2), 3, 'tf_intp_D_axis2 shape must be (1, no_pt, 3)')
        tf.assert_equal(tf.rank(tf_coeff_matrix_first_der), 3, 'tf_coeff_matrix_first_der shape must be (1, 5, 3)')
        tf.assert_equal(tf.rank(tf_coeff_matrix_second_der), 3, 'tf_intp_u_axis1 shape must be (1, 3, 1)')

        tf_h1 = tf.matmul(tf_intp_u_axis1, tf_coeff_matrix_first_der)
        tf_dudx = tf.divide(tf_h1, interpolated_spacing)
        tf_divD_dudx = tf.multiply(tf_dudx, tf_intp_D_axis1)
        tf_h2 = tf.matmul(tf_divD_dudx, tf_coeff_matrix_second_der)
        tf_term1 = tf.divide(tf_h2, interpolated_spacing)

        tf_h1 = tf.matmul(tf_intp_u_axis2, tf_coeff_matrix_first_der)
        tf_dudy = tf.divide(tf_h1, interpolated_spacing)
        tf_divD_dudy = tf.multiply(tf_dudy, tf_intp_D_axis2)
        tf_h2 = tf.matmul(tf_divD_dudy, tf_coeff_matrix_second_der)
        tf_term2 = tf.divide(tf_h2, interpolated_spacing)

        tf_dDdV_dx2 = tf.add(tf_term1, tf_term2)
        tf_dDdV_dx2 = tf.add(tf_dDdV_dx2, tf_weight_c)

        return tf_dDdV_dx2

    def assign_training_loss(self, n_training_epoch, training_loss):
        self.n_training_epoch = n_training_epoch
        self.training_loss = training_loss
        return

    def instance_to_dic(self):
        diffusion_dl_model_instance = \
            {'nn_indices:': self.nn_indices,
             'interpolated_spacing': self.interpolated_spacing,
             'dist_intp_coord_axis1': self.dist_intp_coord_axis1,
             'dist_intp_coord_axis2': self.dist_intp_coord_axis2,
             't': self.t,
             'u': self.u,
             'no_pt': self.no_pt,

             'tf_weight_D': self.tf_weight_D,
             'tf_weight_c': self.tf_weight_c,
             'tf_lowest_weight_D': self.tf_lowest_weight_D,
             'tf_lowest_weight_c': self.tf_lowest_weight_c,

             'nn_u': self.nn_u,
             'nn_weight_D': self.nn_weight_D,

             'tf_intp_u_axis1': self.tf_intp_u_axis1,
             'tf_intp_u_axis2': self.tf_intp_u_axis2,
             'tf_dudt': self.tf_dudt,

             'training_loss': self.training_loss,
             'n_training_epoch': self.n_training_epoch,
             }
        return diffusion_dl_model_instance


    '''
    def del_D_delV_9pt(self, input_to_model, D_tf):
        h1 = tf.matmul(input_to_model, self.first_derivative) / self.interpolated_spacing  # [time_pt, n_coord, 1, 5]
        h2 = tf.multiply(h1, D_tf)  #
        h3 = tf.matmul(h2, self.second_derivative) / self.interpolated_spacing  # [time_pt, n_coord, 1, 1]
        output = tf.add(h3, self.c)  # [time_pt, n_coord, 1, 1]

        return output

    def interpolate_D_9_pt(self, D):
        """
        :param:D tensor
        :return D_tf (1, n_coord, 1, 5) tensor
        """
        assert tf.is_tensor(D) is True, 'D input must be a tensor'

        D = tf.squeeze(self.D)
        nn_D = tf.gather(D, self.nn_indices)

        # ======= intp_D_axis1 and intp_D_axis2 shape expected to be (n_coord, 5) ======
        intp_D_axis1 = self.ut.tf_idw_interpolate(self.dist_intp_coord_axis1, nn_D)
        intp_D_axis2 = self.ut.tf_idw_interpolate(self.dist_intp_coord_axis2, nn_D)

        # intp_D shape expected to be (n_coord, 10)
        intp_D_ = tf.concat([intp_D_axis1, intp_D_axis2], axis=-1)
        assert intp_D_.shape[-1] == 10, 'intp_D shape last dimension not equal to 10 after concatenation'

        # rearrange intp_D_ index
        idx = np.array([7, 3, 1, 8, 6])  # FIXED. to arrange the indices based on location
        intp_D_ = tf.gather(intp_D_, idx, axis=-1)  # at here, intp_D_ should have shape of [n_cood, 5]
        assert np.ndim(intp_D_) == 2, 'intp_D_out should have 2 dimension (n_coord, 5)'
        assert np.shape(intp_D_)[1] == 5, 'intp_D_out first dimension should be 5'

        intp_D_ = tf.expand_dims(intp_D_, 0)  # shape (1, n_coord, 5)
        intp_D_ = tf.expand_dims(intp_D_, 2)  # shape (1, n_coord, 1, 5)
        D_tf = tf.convert_to_tensor(intp_D_, dtype=tf.float64)

        return D_tf
        
    def call_model(self, input_to_model):
        # input is the 1 point has 9 nn_u. expected shape (time_pt, n_coord, 1, 9)
        tf.assert_equal(tf.rank(input_to_model), 4, 'input_to_model should have 3 dimension (time_pt, n_coord,1,9nn_u)')

        D_tf = self.interpolate_D(self.D)  # output shape is (n_coord, 5) - numpy
        output = self.del_D_delV(input_to_model, D_tf)

        return output

    def combine_intp_u_axis1_axis2_network(self):
        print('Compute input for network: Interpolating nn_u for every time step')
        """
        input u is (time_pt, n_coord) numpy
        this function will output u (time_pt, n_coord, 1, 9) tensor
        9 neighbours where 0 is center:
        1 is right, 2 is left, 3 is down, 4 is up
        5 is right right, 6 is left left, 7 is down down, 8 is up up
        """

        intp_u_axis1 = self.compute_intp_u_axis1()
        intp_u_axis2 = self.compute_intp_u_axis2()

        intp_u_ = np.concatenate([intp_u_axis1, intp_u_axis2], axis=-1)
        intp_u_ = np.delete(intp_u_, -1, axis=0)
        assert np.shape(intp_u_)[-1] == 10, 'intp_u shape last dimension not equal to 10 after concatenation'

        # rearrange intp_u index
        idx = np.array([2, 3, 1, 8, 6, 4, 0, 9, 5])
        intp_u_ = intp_u_[:, :, idx]  # shape of (time_pt, n_coord, 9)
        assert np.shape(intp_u_)[-1] == 9, 'intp_u last dimension should be 9'
        model_input_u = np.expand_dims(intp_u_, 2)  # shape of (time_pt, n_coord, 1, 9)
        # self.cc.insert(self.input_model

        return model_input_u

    def assign_input_to_model(self, input_to_model):
        """
        :param input_to_model: list of array
        :return:
        """
        self.intp_u_axis1 = tf.convert_to_tensor(input_to_model, dtype=tf.float64)
        return
        
    def compute_intp_u_axis1(self):
        dist_intp_u_axis1 = self.dist_intp_coord_axis1.copy()
        intp_u_axis1 = []
        nn_u = self.u[:, self.nn_indices]

        for nn_u in nn_u:
            intp_u_axis1_tmp = self.ut.idw_interpolate(dist_intp_u_axis1, nn_u)
            intp_u_axis1.append(intp_u_axis1_tmp)

        return intp_u_axis1

    def compute_intp_u_axis2(self):
        dist_intp_u_axis2 = self.dist_intp_coord_axis2.copy()
        intp_u_axis2 = []
        nn_u = self.u[:, self.nn_indices]

        for nn_u in nn_u:
            intp_u_axis2_tmp = self.ut.idw_interpolate(dist_intp_u_axis2, nn_u)
            intp_u_axis2.append(intp_u_axis2_tmp)

        return intp_u_axis2
    '''


