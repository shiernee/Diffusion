import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import GlobalParameters as gp
from Utils import Utils
import symfit
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from Parameters import Parameters
from InterpolationMethod import InterpolationMethod


class DiffusionModel:
    # ===== this model will compute all the neccesary parameter needed for forward solver====== #

    def __init__(self,  point_cloud, u0, D, c):

        self.p = Parameters()
        self.interpolate = InterpolationMethod()
        self.ut = Utils()
        self.check_inputs(point_cloud, u0, D, c)

        self.point_cloud = point_cloud
        self.u = u0
        self.D = D
        self.c = c

        if self.p.interpolate_method.get('interpolate_type') == 'Rbf':
            intp_function = self.p.interpolate_method.get('function')
            self.intp_D_axis1, self.intp_D_axis2 = self.interpolate_rbf(self.D, intp_function)

        self.t = None
        self.u_exact = None
        self.nn_u = None
        # self.nn_D = None
        # self.nn_u0 = None

    def check_inputs(self, point_cloud, u0, D, c):
        if isinstance(point_cloud, object) is False:
            raise ValueError("point_cloud should be an object")
        if u0.ndim != 1:
            raise ValueError("u0 should be 1D array")
        if D.ndim != 1:
            raise ValueError("D should be 1D array")
        if c.ndim != 1:
            raise ValueError("c should be 1D array")
        return

    # def assign_point_cloud_object(self, point_cloud_obj):
    #     self.point_cloud = point_cloud_obj
    #     return
    #
    # def assign_u0(self, u0):
    #     u0 = np.reshape(u0, [1, self.point_cloud.no_pt])
    #     assert np.ndim(u0) == 2, 'time should be 2D array'
    #     self.u = u0.copy()
    #     return

    def assign_u_update(self, u_update):
        self.u = []
        self.u = u_update
        return

    def assign_u_exact(self, u_exact):
        self.u_exact = u_exact
        return

    def assign_t(self, time_pt):
        self.t = []
        self.t = time_pt
        return

    # def assign_D_c(self, D, c):
    #     assert np.ndim(D) == 1, 'D should be 1D array'
    #     assert np.ndim(c) == 1, 'c should be 1D array'
    #     self.D, self.c = D.copy(), c.copy()
    #     return

    def compute_nn_u(self):
        nn_indices = self.point_cloud.nn_indices.copy()
        u = self.u.copy()

        assert np.ndim(u[:, nn_indices][:5]) == 3, 'nn_u shape must have 3 dimension'
        self.nn_u = u[:, nn_indices]
        return

    # def compute_nn_u0(self):
    #     nn_indices = self.point_cloud.nn_indices.copy()
    #     u0 = self.u0.copy()
    #
    #     assert np.ndim(u0[:, nn_indices][:5]) == 3, 'nn_u shape must have 3 dimension'
    #     self.nn_u0 = u0[:, nn_indices]
    #     return

    # def compute_nn_D(self):
    #     nn_indices = self.point_cloud.nn_indices.copy()
    #     D = self.D.copy()
    #
    #     assert np.ndim(D) == 1, 'D shape must only has 1 dimension'
    #     assert np.ndim(D[nn_indices]) == 2, 'nn_D shape must have 2 dimension'
    #     self.nn_D = D[nn_indices]
    #     return

    # def assign_nn_D(self, nn_D):
    #     self.nn_D = nn_D.copy()
    #     return

    # def interpolate_D_not_used(self, dist_intp_coord_axis, order_acc):
    #     # nn_coord = self.point_cloud.nn_coord.copy()
    #     nn_D = self.nn_D.copy()
    #     D = self.D.copy()
    #
    #     # assert nn_coord is not None, 'nn_coord is None'
    #     assert nn_D is not None, 'nn_D is None'
    #
    #     number_of_pt_to_be_interpolated = order_acc + 2
    #     ind = int(number_of_pt_to_be_interpolated / 2)
    #
    #     intp_D_axis = self.ut.idw_interpolate(dist_intp_coord_axis, nn_D)
    #     # intp_D_axis = self.ut.idw_interpolate(intp_coord_axis, nn_coord, nn_D)
    #     intp_D_axis[:, ind] = D
    #     return intp_D_axis

    # def interpolate_D(self, dist_intp_coord_axis, nn_indices_intp_coord_axis, order_acc):
    #     # nn_coord = self.point_cloud.nn_coord.copy()
    #     n_neighbor = nn_indices_intp_coord_axis.shape[-1]
    #     nn_D = self.D[nn_indices_intp_coord_axis].reshape([-1, n_neighbor])
    #     dist_intp_coord_axis = dist_intp_coord_axis.reshape([-1, n_neighbor])
    #     D = self.D.copy()
    #     # assert nn_coord is not None, 'nn_coord is None'
    #     assert nn_D is not None, 'nn_D is None'
    #
    #     fd_coeff_length = self.ut.fd_coeff_length(order_acc)
    #     number_of_pt_to_be_interpolated = self.ut.compute_no_pt_needed_for_interpolation(fd_coeff_length,
    #                                                                                      order_derivative=2)
    #     ind = int(number_of_pt_to_be_interpolated / 2)
    #
    #     intp_D_axis = self.ut.idw_interpolate(dist_intp_coord_axis, nn_D)
    #     # intp_D_axis = self.ut.idw_interpolate(intp_coord_axis, nn_coord, nn_D)
    #     intp_D_axis = intp_D_axis.reshape([self.point_cloud.no_pt, number_of_pt_to_be_interpolated])
    #     intp_D_axis[:, ind] = D
    #     return intp_D_axis

    # def assign_intp_D_axis1(self, intp_D_axis1):
    #     self.intp_D_axis1 = intp_D_axis1.copy()
    #     return
    #
    # def assign_intp_D_axis2(self, intp_D_axis2):
    #     self.intp_D_axis2 = intp_D_axis2.copy()
    #     return

    # def interpolate_u(self, dt=None, duration=None):
    #     nn_u = self.nn_u.copy()
    #     self.fourier_smooth_next_time_pt_u(self.u, 3)
    #     intp_coord_axis1 = self.intp_coord_axis1.copy()
    #     intp_coord_axis2 = self.intp_coord_axis2.copy()
    #     nn_coord = self.nn_coord.copy()
    #
    #     assert np.ndim(nn_u) == 3, 'nn_u shape must have 3 dimension'
    #
    #     time_pt = np.linspace(0, duration, int(duration / dt) + 1, endpoint=True, dtype='float64')
    #     dist_intp_u_axis1 = self.point_cloud.dist_intp_coord_axis1.copy()
    #     dist_intp_u_axis2 = self.point_cloud.dist_intp_coord_axis2.copy()
    #     intp_u_axis1 = []
    #     intp_u_axis2 = []
    #
    #     for t in range(1, len(time_pt)):
    #         # intp_u_axis1.append(self.ut.idw_interpolate(intp_coord_axis1,
    #         #                                             nn_coord, nn_u[t - 1]))
    #         # intp_u_axis2.append(self.ut.idw_interpolate(intp_coord_axis2,
    #         #                                             nn_coord, nn_u[t - 1]))
    #
    #         intp_u_axis1.append(self.ut.idw_interpolate(dist_intp_u_axis1, nn_u[t-1]))
    #         intp_u_axis2.append(self.ut.idw_interpolate(dist_intp_u_axis2, nn_u[t-1]))
    #     intp_u_axis1 = np.array(intp_u_axis1, dtype='float64')
    #     intp_u_axis2 = np.array(intp_u_axis2, dtype='float64')
    #
    #     return intp_u_axis1, intp_u_axis2

    def del_D_delV_method1(self, u, interpolated_spacing, order_acc, coeff_matrix_first_der, coeff_matrix_second_der):
        assert np.ndim(u) == 1, 'u must be 1D array'

        c = self.c.copy()
        intp_D_axis1 = self.intp_D_axis1
        intp_D_axis2 = self.intp_D_axis2

        ################ INTERPOLATION U ################
        '''
        # ==== idw interpolation ======
        n_neighbor = self.point_cloud.nn_indices_intp_coord_axis1.shape[-1]
        fd_coeff_length = self.ut.fd_coeff_length(order_acc)
        number_of_pt_to_be_interpolated = self.ut.compute_no_pt_needed_for_interpolation(fd_coeff_length,
                                                                                         order_derivative=2)

        # METHOD 1: ==== idw_interpolate with high V0 value first, include interpolated point to get nn =====
        points_to_interpolate = np.concatenate((self.point_cloud.intp_coord_axis1.reshape([-1, 3]),
                                                self.point_cloud.intp_coord_axis2.reshape([-1, 3])),
                                                axis=0)
        intp_u_axis1, intp_u_axis2 = self.ut.idw_interpolate_sort(self.point_cloud.coord, u, points_to_interpolate)

        # METHOD 2: === interpolate intp_u independently using individual neighbour ===
        ind = int(number_of_pt_to_be_interpolated / 2)
        nn_u_axis1 = u[self.point_cloud.nn_indices_intp_coord_axis1].reshape([-1, n_neighbor])
        nn_u_axis2 = u[self.point_cloud.nn_indices_intp_coord_axis2].reshape([-1, n_neighbor])
        dist_intp_coord_axis1 = self.point_cloud.dist_intp_coord_axis1.reshape([-1, n_neighbor])
        dist_intp_coord_axis2 = self.point_cloud.dist_intp_coord_axis2.reshape([-1, n_neighbor])
        intp_u_axis1 = self.ut.idw_interpolate(dist_intp_coord_axis1, nn_u_axis1)
        intp_u_axis2 = self.ut.idw_interpolate(dist_intp_coord_axis2, nn_u_axis2)
        intp_u_axis1 = intp_u_axis1.reshape([self.point_cloud.no_pt, number_of_pt_to_be_interpolated])
        intp_u_axis2 = intp_u_axis2.reshape([self.point_cloud.no_pt, number_of_pt_to_be_interpolated])
        intp_u_axis1[:, ind] = u.copy()
        intp_u_axis2[:, ind] = u.copy()
        # METHOD3: ===============================================================
        nn_u = u[self.point_cloud.nn_indices]
        intp_u_axis1 = self.ut.idw_interpolate(self.point_cloud.dist_intp_coord_axis1, nn_u)
        intp_u_axis2 = self.ut.idw_interpolate(self.point_cloud.dist_intp_coord_axis2, nn_u)

        # METHOD 4:=== fourier fit first then use the smooth model to get intp_u
        fit_fourier_smooth_model = self.fourier_smooth_u(u, order_fourier_fit=5)
        intp_u_axis1, intp_u_axis2 = self.interpolate_u_fourier_model(fit_fourier_smooth_model)
        '''
        # METHOD5:
        intp_u_axis1, intp_u_axis2 = [], []
        for i in range(self.point_cloud.no_pt):
            nn_coord = self.point_cloud.nn_coord[i]
            nn_u = u[self.point_cloud.nn_indices[i]]
            r, phi, theta = self.ut.xyz2sphr(nn_coord)

            intp_coord = np.concatenate((self.point_cloud.intp_coord_axis1[i],
                                         self.point_cloud.intp_coord_axis2[i]), axis=0)
            rintp, phiintp, thetaintp = self.ut.xyz2sph(intp_coord)

            rbfi_s = self.interpolate(r, phi, theta, nn_u)
            intp_var_u = rbfi_s(rintp, phiintp, thetaintp)
            intp_var_u = np.array(intp_var_u, dtype='float64').reshape(([-1, 2]))
            intp_u_axis1, intp_u_axis2 = intp_var_u[:, 0], intp_var_u[:, 1]


        # intp_u_axis1, intp_u_axis2 = self.interpolate(u)
        #########################################################################

        assert np.shape(intp_u_axis1)[-1] == np.shape(coeff_matrix_first_der)[0], \
            print('intp_u_axis1 shape of {} not match with coeff_matrix shape {}'.
                  format(np.shape(intp_u_axis1), np.shape(coeff_matrix_first_der)))
        assert np.shape(intp_u_axis2)[-1] == np.shape(coeff_matrix_first_der)[0], \
            print('intp_u_axis2 shape of {} not match with coeff_matrix shape {}'.
                  format(np.shape(intp_u_axis2), np.shape(coeff_matrix_second_der)))

        fd_coeff_length = order_acc + 1
        shape_to_be_taken = np.array([np.floor(np.shape(intp_u_axis1)[-1] / 2) - np.floor(fd_coeff_length / 2),
                                      np.floor(np.shape(intp_u_axis1)[-1] / 2) + np.floor(fd_coeff_length / 2) + 1],
                                     dtype='int')

        # # ==== method 1 ======
        dudx = np.matmul(intp_u_axis1, coeff_matrix_first_der) / interpolated_spacing
        dudy = np.matmul(intp_u_axis2, coeff_matrix_first_der) / interpolated_spacing

        if self.bc.bc_type == 'neumann':
            dudx, dudy = self.bc.call_bc(dudx, dudy)
        if self.bc.bc_type == 'periodic':
            dudx, dudy = self.bc.call_bc(dudx, dudy)

        divD_dudx = np.multiply(dudx, intp_D_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]])
        divD_dudy = np.multiply(dudy, intp_D_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]])

        term1 = np.matmul(divD_dudx, coeff_matrix_second_der) / interpolated_spacing
        term2 = np.matmul(divD_dudy, coeff_matrix_second_der) / interpolated_spacing

        dDdV_dx2 = np.squeeze(term1 + term2) + c

        # ===== method 3 ======
        # coeff_matrix = np.zeros([intp_u_axis1.shape[1], 1])
        # mid = int(intp_u_axis1.shape[1] / 2)
        # coeff = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]).reshape([-1, 1])
        # coeff_matrix[mid - int(len(coeff) / 2):mid + int(len(coeff) / 2) + 1] = coeff
        #
        # D_prime_dx = np.matmul(intp_D_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]],
        #                        coeff_matrix_second_der) / interpolated_spacing
        # D_prime_dy = np.matmul(intp_D_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]],
        #                        coeff_matrix_second_der) / interpolated_spacing
        # num_u_prime_dx = np.matmul(intp_u_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]],
        #                            coeff_matrix_second_der) / interpolated_spacing
        # num_u_prime_dy = np.matmul(intp_u_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]],
        #                            coeff_matrix_second_der) / interpolated_spacing
        # num_u_pp_dx = np.matmul(intp_u_axis1, coeff_matrix) / interpolated_spacing
        # num_u_pp_dy = np.matmul(intp_u_axis2, coeff_matrix) / interpolated_spacing
        #
        # D = self.D.reshape([-1, 1])
        # num_term1_3 = D_prime_dx * num_u_prime_dx + D_prime_dy * num_u_prime_dy
        # num_term2_3 = D * num_u_pp_dx + D * num_u_pp_dy
        # dDdV_dx2 = np.squeeze(num_term1_3 + num_term2_3) + c

        return dDdV_dx2



    # def fourier_smooth_u(self, u, order_fourier_fit=3):
    #     coord = self.point_cloud.coord.copy()
    #     _, phi_data, theta_data = self.ut.xyz2sph(coord)
    #     u_data = np.squeeze(u)
    #
    #     phi, theta, u_fit = symfit.variables('phi, theta, u_fit')
    #     w, = symfit.parameters('w')
    #     model_dict = {u_fit: self.ut.two_variable_fourier_series(phi, theta, f=w, n=order_fourier_fit)}
    #     fit_fourier_smooth_model = symfit.Fit(model_dict, phi=phi_data, theta=theta_data, u_fit=u_data)
    #
    #     fit_result = fit_fourier_smooth_model.execute()
    #     print(fit_result)
    #     import matplotlib.pyplot as plt
    #
    #     ind = np.argsort(theta_data)
    #     plt.plot(theta_data[ind], np.squeeze(self.u0)[ind])
    #     plt.plot(theta_data[ind], u_data[ind], ls='-')
    #     plt.plot(theta_data[ind], fit_fourier_smooth_model.model(phi=phi_data, theta=theta_data,
    #                                                              **fit_result.params).u_fit, ls=':')
    #     plt.show()
    #     return fit_fourier_smooth_model
    #
    # def interpolate_u_fourier_model(self, fit_fourier_smooth_model):
    #     intp_u_axis1 = []
    #     fit_result = fit_fourier_smooth_model.execute()
    #     for coord in self.point_cloud.intp_coord_axis1:
    #         _, phi_data_tmp, theta_data_tmp = self.ut.xyz2sph(coord)
    #         intp_u_axis1.append(fit_fourier_smooth_model.model(phi=phi_data_tmp, theta=theta_data_tmp,
    #                                                            **fit_result.params).u_fit)
    #
    #     intp_u_axis2 = []
    #     for coord in self.point_cloud.intp_coord_axis2:
    #         _, phi_data_tmp, theta_data_tmp = self.ut.xyz2sph(coord)
    #         intp_u_axis2.append(fit_fourier_smooth_model.model(phi=phi_data_tmp, theta=theta_data_tmp,
    #                                                            **fit_result.params).u_fit)
    #
    #     return intp_u_axis1, intp_u_axis2

    def instance_to_dict(self):
        physics_model_instance = \
            {'t': self.t,
             'u0': self.u0,
             'u': self.u,
             'u_exact': self.u_exact,
             'D': self.D,
             'c': self.c,
             'nn_u0': self.nn_u0,
             'nn_u': self.nn_u,
             'nn_D': self.nn_D,
             'intp_D_axis1': self.intp_D_axis1,
             'intp_D_axis2': self.intp_D_axis2,
             }

        return physics_model_instance

    def assign_read_physics_model_instances(self, physics_model_instances):
        self.t = physics_model_instances['t']
        self.u0 = physics_model_instances['u0']
        self.u = physics_model_instances['u']
        self.u_exact = physics_model_instances['u_exact']
        self.D = physics_model_instances['D']
        self.c = physics_model_instances['c']
        self.nn_u0 = physics_model_instances['nn_u0']
        self.nn_u = physics_model_instances['nn_u']
        self.nn_D = physics_model_instances['nn_D']
        self.intp_D_axis1 = physics_model_instances['intp_D_axis1']
        self.intp_D_axis2 = physics_model_instances['intp_D_axis2']
        print('Finish assigning read instances to Physics_Model instances')
        return


    '''
    def coord_fit_to_plane(self, coord):
        # fit a plane (Ax + By + Cz + D = 0) using coordinates
        # a0 = A/C ; a1 = B/C; a2 = D/C
        # fit = [a0, a1, a2]
        tmp_A = []
        tmp_b = []
        for i in range(len(coord)):
            tmp_A.append([coord[i, 0], coord[i, 1], 1])
            tmp_b.append(coord[i, 2])
        b = np.array(tmp_b).T
        A = np.array(tmp_A)
        fit = np.linalg.inv(A.T @ A) @ A.T @ b
        bfit = (A @ fit).reshape([-1, 1])
        coord_on_plane = np.concatenate((coord[:, :2], bfit), axis=1)
        return fit, coord_on_plane

    def projected_pt_on_plane(self, three_coord_on_plane, coord_to_be_projected):
        # compute projected coordinate using coordinates on the plane and coordinates
        
        assert three_coord_on_plane.shape[0] == 3, 'coordinates passed in is not three points'

        # project points onto the plane
        coord1_on_plane = three_coord_on_plane[0]
        local_origin = three_coord_on_plane[1]
        coord3_on_plane = three_coord_on_plane[2]

        v_a = coord1_on_plane - local_origin
        v_b = coord3_on_plane - local_origin
        v_n = np.cross(v_a, v_b)

        v_n = v_n / np.linalg.norm(v_n)  # unit vector of normal vector to the plane
        v_ = coord_to_be_projected - local_origin

        distance = np.sum(v_ * v_n, axis=1, keepdims=True)
        projected_coord = coord_to_be_projected - v_n * distance
        shortest_distance = np.linalg.norm(projected_coord - coord_to_be_projected, axis=1)
        return projected_coord, shortest_distance

    def local_xy_coord(self, projected_coord, local_origin, local_x_axis, local_y_axis):
        # compute local coordinate using local coordinate system
        # example:
        # projected_coord = coordinate projected on the plane using function projected_pt_on_plane, these points are not
        # perpendicular projected to the plane
        # local_origin = (0, 0, 0)
        # local_x_axis = new local x_axis on the plane
        # local_y_axis = new local y_axis on the plane
        
        v_ = projected_coord - local_origin
        local_coord = []
        for i in range(len(v_)):
            local_x_coord = np.dot(v_[i], local_x_axis)
            local_y_coord = np.dot(v_[i], local_y_axis)
            local_coord.append([local_x_coord, local_y_coord])
        local_coord = np.array(local_coord, dtype='float64')
        return local_coord

    def rotation_matrix(self, ori_coord, final_coord):
        # find the rotatin matrix to rotate coordinates at original coordinate system to final coordinate system
        # example:
        # ori_origin = np.array([0, 0, 0])
        # ori_x_axis = [1, 0, 0]
        # ori_y_axis = [0, 1, 0]
        # ori_z_axis = [0, 0, 1]
        # ori_coord_sys = np.array([ori_x_axis, ori_y_axis, ori_z_axis])
        # 
        # final_x_axis = [2, 4 ,5]
        # final_y_axis = [5, 4 ,5]
        # final_z_axis = [6, 4 ,5]
        # final_coord_sys = np.array([final_x_axis, final_y_axis, final_z_axis])
        

        if np.ndim(ori_coord) != 2:
            raise Exception('original coordinate system must be 2D array')
        if np.ndim(final_coord) != 2:
            raise Exception('final coordinate system must be 2D array')
        if np.shape(ori_coord)[0] != 3 or np.shape(ori_coord)[1] != 3:
            raise Exception('original coordinate system must be 3x3 array')
        if np.shape(final_coord)[0] != 3 or np.shape(final_coord)[1] != 3:
            raise Exception('original coordinate system must be 3x3 array')

        M11 = np.dot(ori_coord[0], final_coord[0])
        M12 = np.dot(ori_coord[0], final_coord[1])
        M13 = np.dot(ori_coord[0], final_coord[2])
        M21 = np.dot(ori_coord[1], final_coord[0])
        M22 = np.dot(ori_coord[1], final_coord[1])
        M23 = np.dot(ori_coord[1], final_coord[2])
        M31 = np.dot(ori_coord[2], final_coord[0])
        M32 = np.dot(ori_coord[2], final_coord[1])
        M33 = np.dot(ori_coord[2], final_coord[2])
        R = np.array([[M11, M12, M13],
                      [M21, M22, M23],
                      [M31, M32, M33]])
        return R

    def glob2local_R(self, coord, R, ori_origin, final_origin):
        assert np.ndim(coord) == 2, 'coordinate to be changed should be 2D array'
        assert np.shape(coord)[1] == 3, 'coordinate to be changed should have 3 columns'
        assert R.shape[0] == R.shape[1], 'R matrix is not symmetric'
        assert np.ndim(ori_origin) == 1, 'original coordinate should be 1D array'
        assert np.ndim(final_origin) == 1, 'original coordinate should be 1D array'

        final_origin_ = np.matmul(final_origin, R)
        translation = ori_origin - final_origin_
        coordinate_at_local_coordinate_system = []
        for n in range(coord.shape[0]):
            coordinate_at_local_coordinate_system.append(np.matmul(coord[n], R) + translation)

        coordinate_at_local_coordinate_system = np.array(coordinate_at_local_coordinate_system, dtype='float')

        return coordinate_at_local_coordinate_system

    def check_glob2local_methods_comparison(self, coord):

        ori_origin = np.array([0, 0, 0])
        ori_x_axis = [1, 0, 0]
        ori_y_axis = [0, 1, 0]
        ori_z_axis = [0, 0, 1]
        ori_coord_sys = np.array([ori_x_axis, ori_y_axis, ori_z_axis])

        _, coord_on_plane = self.coord_fit_to_plane(coord)
        projected_coord, _ = self.projected_pt_on_plane(coord_on_plane[:3, :], coord)
        nn_coord = np.expand_dims(projected_coord, 0)
        local_origin = np.expand_dims(coord[0], 0)
        local_x_axis, local_y_axis = self.compute_local_axis(nn_coord, local_origin)
        local_x_axis, local_y_axis = np.squeeze(local_x_axis), np.squeeze(local_y_axis)
        local_z_axis = np.cross(local_x_axis, local_y_axis)
        final_coord_sys = np.array([local_x_axis, local_y_axis, local_z_axis])

        R = self.rotation_matrix(ori_coord_sys, final_coord_sys)

        coord1 = self.glob2local_map(coord)
        coord2 = self.glob2local_R(coord, R, ori_origin, final_origin=coord[0])
        error = np.sum(coord1 - coord2[:, :2])

        if error > 1e-3:
            raise Exception('local coordinates computed using different methods did not match')
        print('Local coordinates computation - PASS')
    '''

    def interpolate_rbf(self, var):
        intp_var_axis1, intp_var_axis2 = [], []

        for i in range(self.point_cloud.no_pt):
            nn_coord = self.point_cloud.nn_coord[i]
            r, phi, theta = self.ut.xyz2sph(nn_coord)
            nn_var = var[self.point_cloud.nn_indices[i]]

            intp_coord = np.concatenate((self.point_cloud.intp_coord_axis1[i], self.point_cloud.intp_coord_axis2[i]),
                                        axis=0)
            rintp, phiintp, thetaintp = self.ut.xyz2sph(intp_coord)

            rbfi_s = Rbf(r, phi, theta, nn_var, function=intp_function)
            intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)

            intp_var_axis1.append(intp_var_nn_s[:int(len(intp_coord) / 2)])
            intp_var_axis2.append(intp_var_nn_s[int(len(intp_coord) / 2):])

        intp_var_axis1 = np.array(intp_var_axis1, dtype='float64')
        intp_var_axis2 = np.array(intp_var_axis2, dtype='float64')
        return intp_var_axis1, intp_var_axis2
