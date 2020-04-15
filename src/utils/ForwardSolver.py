import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
from Utils import Utils


class ForwardSolver:
    def __init__(self, point_cloud, physics_model, interpolated_spacing, order_acc):
        self.point_cloud = point_cloud
        self.physics_model = physics_model

        self.interpolated_spacing = interpolated_spacing
        self.order_acc = order_acc
        self.ut = Utils()

        self.coeff_matrix_first_der = None
        self.coeff_matrix_second_der = None
        return

    def solve(self, dt, duration):
        assert self.physics_model.nn_u0.shape[0] == 1, 'first axis of nn_u must be 1, indicating nn_u0'
        assert self.physics_model.u0.shape[0] == 1, 'first axis of u must be 1, indicating u0'

        coeff_matrix_first_der = self.coeff_matrix_first_der.copy()
        coeff_matrix_second_der = self.coeff_matrix_second_der.copy()
        u = np.squeeze(self.physics_model.u0).copy()
        u0 = np.squeeze(self.physics_model.u0).copy()

        time_pt = np.linspace(0, duration, int(duration/dt)+1, endpoint=True, dtype='float64')

        u_update = []
        u_update.append(u)
        time_pt_saved = []
        time_pt_saved.append(0)
        time_tmp = 0.1
        disp_time_tmp = 0.001
        for t in range(1, len(time_pt)):
            # print('{}time: {}'.format(t, t*dt))

            deltaD_deltaV = self.physics_model.del_D_delV(u, self.interpolated_spacing, self.order_acc, \
                                                          coeff_matrix_first_der, coeff_matrix_second_der)
            dudt = deltaD_deltaV.copy()
            next_time_pt_u = u + dt * dudt

            assert np.max(next_time_pt_u) <= np.max(u0)*1.1, \
                'at time {:.2f}, next_time_pt_u {:.2f} higher than 10% of initial highest temperature, {}'.\
                    format(t*dt, np.max(next_time_pt_u), np.max(self.physics_model.u0))

            assert np.min(next_time_pt_u) >= np.min(u0)*1.1, \
                'at time {:.2f}, next_time_pt_u {:.2f} lower than 10% ofinitial lowest temperature, {}'.\
                    format(t*dt, np.min(next_time_pt_u), np.min(self.physics_model.u0))

            u = next_time_pt_u.copy()

            if (abs((t * dt) - time_tmp)) < 1e-6:
                print('====================================================')
                print('time {}'.format(t * dt))
                disp_time_tmp += 0.001

            if (abs((t * dt) - time_tmp)) < 1e-6:
                print('====================================================')
                print('saving at time {}'.format(t * dt))
                time_pt_saved.append(time_tmp)
                u_update.append(u)
                time_tmp += 0.1

        u_update = np.array(u_update, dtype='float64')
        time_pt_saved = np.array(time_pt_saved, dtype='float64')
        return u_update, time_pt_saved

    def generate_first_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length = np.shape(self.point_cloud.intp_coord_axis1)[1]
        coeff_matrix_first_der = self.ut.coeff_matrix_first_order(input_length, coeff)
        self.coeff_matrix_first_der = coeff_matrix_first_der.copy()
        return

    def generate_second_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length2 = np.shape(self.point_cloud.intp_coord_axis1)[1] - len(coeff) + 1
        coeff_matrix_second_der = self.ut.coeff_matrix_first_order(input_length2, coeff)
        self.coeff_matrix_second_der = coeff_matrix_second_der.copy()
        return





