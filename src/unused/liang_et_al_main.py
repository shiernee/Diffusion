import sys
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\FHN\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\FileIO\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Utils\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\PointCloud\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Diffusion\src\\utils')

import numpy as np
from FileIO import FileIO
from GeneratePoints import GeneratePoints
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import Utils
from PointCloud import PointCloud
from DiffusionModel import DiffusionModel
from ForwardSolver import ForwardSolver
from DiffusionDLModel import DiffusionDLModel
from Parameters import Parameters
from GenerateDC import GenerateDC
from InterpolatedSpacing import InterpolatedSpacing


#  ==== liang et al study =====

if __name__ == '__main__':

    p = Parameters()

    np.random.seed(7892)
    fileio = FileIO()
    fileio.assign_forward_folder(p.forward_folder)

    gen_pt = GeneratePoints()
    cart_coord, sph_coord = gen_pt.generate_points_sphere(no_pt=p.no_pt, sph_radius=1)
    r, phi, theta = sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2]
    p.set_u_initial(r=r, phi=phi, theta=theta)
    p.set_interpolated_spacing(cart_coord=cart_coord)

    ut = Utils()
    gen_DC = GenerateDC(no_pt=p.no_pt)
    gen_DC.set_fixed_D(p.D)

    # ========================= compute parameters in physics_model =============================== #
    point_cloud = PointCloud()
    point_cloud.assign_coord(cart_coord)
    point_cloud.compute_nn_indices_neighbor(n_neighbors=p.nn_method.get('n_neighbors'),
                                            algorithm=p.nn_method.get('nn_algorithm'))
    point_cloud.compute_nn_coord()
    point_cloud.compute_local_axis()
    point_cloud.interpolate_coord_local_axis1_axis2(interpolated_spacing=p.interpolated_spacing,
                                                    order_acc=p.order_acc, order_derivative=p.order_derivative,
                                                    clip=p.clip_bool)
    '''
    _, dist_intp_coord_axis1, nn_indices_intp_coord_axis1 = \
        point_cloud.compute_nn_indices_neighbor_intp_coord_axis1(n_neighbors=15, algorithm='kd_tree')
    _, dist_intp_coord_axis2, nn_indices_intp_coord_axis2 = \
        point_cloud.compute_nn_indices_neighbor_intp_coord_axis2(n_neighbors=15, algorithm='kd_tree')
    dist_intp_coord_axis1 \
        = point_cloud.discard_nn_coord_out_of_radius(dist_intp_coord_axis1, radius=3*avg_dx)
    dist_intp_coord_axis2 \
        = point_cloud.discard_nn_coord_out_of_radius(dist_intp_coord_axis2, radius=3*avg_dx)
    
    point_cloud.assign_dist_intp_coord_axis12(dist_intp_coord_axis1, dist_intp_coord_axis2)
    point_cloud.assign_nn_indices_intp_coord_axis12(nn_indices_intp_coord_axis1, nn_indices_intp_coord_axis2)
    '''

    # =================== Diffusion Model ============
    physics_model = DiffusionModel(point_cloud, p.u0, gen_DC.D, gen_DC.c)
    # physics_model.assign_point_cloud_object()
    # physics_model.assign_u0(p.u0)
    # physics_model.assign_D_c(gen_DC.D, gen_DC.c)

    # physics_model.compute_nn_u0()
    # physics_model.compute_nn_D()

    # intp_D_axis1 = physics_model.interpolate_D(point_cloud.dist_intp_coord_axis1,
    #                                            point_cloud.nn_indices_intp_coord_axis1, ORDER_ACC)
    # intp_D_axis2 = physics_model.interpolate_D(point_cloud.dist_intp_coord_axis2,
    #                                            point_cloud.nn_indices_intp_coord_axis2, ORDER_ACC)

    # intp_D_axis1, intp_D_axis2 = physics_model.interpolate_rbf(physics_model.D)

    # intp_D_axis1, intp_D_axis2 = np.ones([point_cloud.no_pt, point_cloud.intp_coord_axis1.shape[1]]), \
    #                              np.ones([point_cloud.no_pt, point_cloud.intp_coord_axis1.shape[1]])
    #
    # physics_model.assign_intp_D_axis1(intp_D_axis1)
    # physics_model.assign_intp_D_axis2(intp_D_axis2)


    # ============================ numpy solver ===================================================== #
    import time
    start = time.time()
    solver = ForwardSolver(point_cloud, physics_model, interpolated_spacing=p.interpolated_spacing,
                           order_acc=p.order_accu)
    solver.generate_first_der_coeff_matrix()
    solver.generate_second_der_coeff_matrix()
    u_update, u_exact, time_pt = solver.solve(dt=p.dt, duration=p.duration)
    end = time.time()
    print('time used:{}'.format(end - start))

    # ============================ tensorflow solver ===================================================== #
    # diffusion_dl_model = DiffusionDLModel()
    # nn_indices = point_cloud.nn_indices
    # diffusion_dl_model.assign_nn_indices(nn_indices)
    # diffusion_dl_model.assign_interpolated_spacing(INTERPOLATED_SPACING)
    # diffusion_dl_model.assign_dist_intp_coord_axis1(dist_intp_coord_axis1)
    # diffusion_dl_model.assign_dist_intp_coord_axis2(dist_intp_coord_axis2)
    # diffusion_dl_model.assign_t(t)
    # diffusion_dl_model.assign_u(u0)
    # diffusion_dl_model.compute_no_pt()
    #
    # inverse_solver = InverseSolver(diffusion_dl_model, ORDER_ACC)
    # tf_intp_u_axis1 = diffusion_dl_model.tf_intp_u_axis1.__copy__()
    # tf_intp_u_axis2 = diffusion_dl_model.tf_intp_u_axis2.__copy__()
    # coeff_matrix_first_der = inverse_solver.generate_first_der_coeff_matrix()
    # coeff_matrix_second_der = inverse_solver.generate_second_der_coeff_matrix()
    #
    # u = u0.copy()
    # for t in range(1, len(time_pt)):
    #     print('time: {}'.format(t * dt))
    #     tf_intp_u_axis1_tmp = tf.slice(tf_intp_u_axis1, begin, size)
    #     tf_intp_u_axis2_tmp = tf.slice(tf_intp_u_axis2, begin, size)
    #
    #     begin = [iteration * batch_size, 0, 0]
    #     size = [batch_size, tf_dudt.shape[1], tf_dudt.shape[2]]
    #     tf_dudt_tmp = tf.slice(tf_dudt, begin, size)
    #
    #     deltaD_deltaV = diffusion_dl_model.call_dl_model(tf_intp_u_axis1_tmp, tf_intp_u_axis2_tmp,
    #                                                      tf_coeff_matrix_first_der,
    #                                                      tf_coeff_matrix_second_der, self.order_acc)
    #
    #     dudt = deltaD_deltaV.copy()
    #     next_time_pt_u = u + dt * dudt
    #
    # # diffusion_dl_model.compute_nn_u()
    #
    # intp_u_axis1, intp_u_axis2 = diffusion_dl_model.interpolate_u_axis1_axis2(ORDER_ACC)
    # diffusion_dl_model.assign_intp_u_axis1(intp_u_axis1)
    # diffusion_dl_model.assign_intp_u_axis2(intp_u_axis2)
    #
    # diffusion_dl_model.initialize_weight()
    # dudt = diffusion_dl_model.compute_dudt()
    # diffusion_dl_model.assign_dudt(dudt)


    # ============================================================================================ #
    physics_model.assign_u_update(u_update)
    physics_model.assign_u_exact(u_exact)
    physics_model.assign_t(time_pt)
    physics_model.compute_nn_u()
    physics_model_instances = physics_model.instance_to_dict()

    point_cloud_instances = point_cloud.instance_to_dict()

    # ================ WRITE DOWN THE PARAMETER USED AND U_UPDATE INTO SAV ===================== #
    i = fileio.file_number_forward_README()
    fileio.write_physics_model_instance(physics_model_instances, i, model='diffusion_liang')
    fileio.write_point_cloud_instance(point_cloud_instances, i)

    with open('{}/{}{}.txt'.format(forward_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('dt={}\n'.format(p.dt))
        csv_file.write('simulation_duration={}\n'.format(p.duration))
        csv_file.write('interpolated_spacing={}\n'.format(p.interpolated_spacing))
        csv_file.write('order_acc={}'.format(p.order_acc))
    print('writing {}/{}{}.txt'.format(forward_folder, 'README', i))


