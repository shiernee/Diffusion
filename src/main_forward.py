import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')


import pickle
from FileIO import FileIO
from DiffusionModel import DiffusionModel
from ForwardSolver import ForwardSolver
from SanityCheck import SanityCheck
from Utils import Utils
from PointCloud import PointCloud
import numpy as np

if __name__=='__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case2_sphere_D(diff)_c0/forward/'

    DT = 5e-5
    DURATION = 0.1
    INTERPOLATED_SPACING = 0.01
    ORDER_ACC = 4  # central difference by taking two elements right and left each to compute gradient
    ORDER_DERIVATIVE = 2   # diffusion term is second derivative

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    sanity_check = SanityCheck()
    ut = Utils()

    # ================= read instances ==============
    instances = fileio.read_generated_instance()
    coord = instances['coord']
    t = np.squeeze(instances['t'])
    u0 = np.squeeze(instances['u'])
    D = np.squeeze(instances['D'])
    c = np.squeeze(instances['c'])

    # ========================= compute parameters in physics_model =============================== #
    point_cloud = PointCloud()
    point_cloud.assign_coord(coord, t)
    point_cloud.compute_no_pt()
    point_cloud.compute_nn_indices_neighbor(n_neighbors=8, algorithm='kd_tree')
    point_cloud.compute_nn_coord()
    point_cloud.compute_local_axis()
    point_cloud.interpolate_coord_local_axis1_axis2(interpolated_spacing=INTERPOLATED_SPACING, order_acc=ORDER_ACC, \
                                                    order_derivative=ORDER_DERIVATIVE)
    _, dist_intp_coord_axis1, nn_indices_intp_coord_axis1 = \
        point_cloud.compute_nn_indices_neighbor_intp_coord_axis1(n_neighbors=24, algorithm='kd_tree')
    _, dist_intp_coord_axis2, nn_indices_intp_coord_axis2 = \
        point_cloud.compute_nn_indices_neighbor_intp_coord_axis2(n_neighbors=24, algorithm='kd_tree')
    dist_intp_coord_axis1 \
        = point_cloud.discard_nn_coord_out_of_radius(dist_intp_coord_axis1, radius=3)
    dist_intp_coord_axis2 \
        = point_cloud.discard_nn_coord_out_of_radius(dist_intp_coord_axis2, radius=3)
    point_cloud.assign_dist_intp_coord_axis12(dist_intp_coord_axis1, dist_intp_coord_axis2)
    point_cloud.assign_nn_indices_intp_coord_axis12(nn_indices_intp_coord_axis1, nn_indices_intp_coord_axis2)

    # point_cloud.compute_distance_intp_coord_axis12_to_ori_coord()
    # point_cloud.compute_dist_nn_to_interpolated_coord_local

    # =================== Diffusion Model ============
    physics_model = DiffusionModel()
    physics_model.assign_point_cloud_object(point_cloud)
    physics_model.assign_u0(u0)
    physics_model.assign_D_c(D, c)
    physics_model.compute_nn_u0()
    physics_model.compute_nn_D()

    intp_D_axis1 = physics_model.interpolate_D(point_cloud.dist_intp_coord_axis1,
                                               point_cloud.nn_indices_intp_coord_axis1, ORDER_ACC)
    intp_D_axis2 = physics_model.interpolate_D(point_cloud.dist_intp_coord_axis2,
                                               point_cloud.nn_indices_intp_coord_axis2, ORDER_ACC)
    physics_model.assign_intp_D_axis1(intp_D_axis1)
    physics_model.assign_intp_D_axis2(intp_D_axis2)

    # ============================================================================================ #

    solver = ForwardSolver(point_cloud, physics_model, interpolated_spacing=INTERPOLATED_SPACING, order_acc=ORDER_ACC)
    solver.generate_first_der_coeff_matrix()
    solver.generate_second_der_coeff_matrix()
    u_update, time_pt = solver.solve(dt=DT, duration=DURATION)

    # ============================================================================================ #
    physics_model.assign_u_update(u_update)
    physics_model.assign_t(time_pt)
    physics_model.compute_nn_u()
    physics_model_instances = physics_model.instance_to_dict()

    point_cloud_instances = point_cloud.instance_to_dict()

    # ================ WRITE DOWN THE PARAMETER USED AND U_UPDATE INTO SAV ===================== #
    i = fileio.file_number_forward_README()
    fileio.write_physics_model_instance(physics_model_instances, i, model='diffusion')
    fileio.write_point_cloud_instance(point_cloud_instances, i)

    with open('{}/{}{}.txt'.format(forward_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('dt={}\n'.format(DT))
        csv_file.write('simulation_duration={}\n'.format(DURATION))
        csv_file.write('interpolated_spacing={}\n'.format(INTERPOLATED_SPACING))
        csv_file.write('order_acc={}'.format(ORDER_ACC))
    print('writing {}/{}{}.txt'.format(forward_folder, 'README', i))



