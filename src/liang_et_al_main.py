import sys
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\FHN\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\FileIO\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Utils\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\PointCloud\src\\utils')
sys.path.insert(1, 'C:\\Users\sawsn\Desktop\Shiernee\\Diffusion\src\\utils')

import numpy as np
from FileIO import FileIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import Utils
from PointCloud import PointCloud
from DiffusionModel import DiffusionModel, BoundaryCondition
from ForwardSolver import ForwardSolver
from DiffusionDLModel import DiffusionDLModel


#  ==== liang et al study =====



def compute_dx(coord):
    dx = np.zeros([len(coord), len(coord)])
    for i in range(len(coord)):
        tmp = coord[i]
        dx[i, :] = np.linalg.norm(tmp - coord, axis=1)

    return np.min(dx[np.nonzero(dx)])


if __name__ == '__main__':

    np.random.seed(7891)
    # ===== generate points on sphere ================
    ut = Utils()
    sph_radius = 1.0
    no_pt = 1002

    phi = np.random.uniform(-np.pi, np.pi, no_pt)
    theta = np.random.uniform(0, np.pi, no_pt)
    radius = np.ones([no_pt, ]) * sph_radius

    sph_coord = np.zeros([no_pt, 3])
    sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2] = radius, phi, theta
    x, y, z = ut.sph2xyz(sph_coord)

    coord = np.zeros([no_pt, 3])
    coord[:, 0], coord[:, 1], coord[:, 2] = x, y, z
    r, phi, theta = sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2]

    duration = 1
    interpolated_spacing = np.float64(0.0025) #compute_dx(coord)
    dt = round(0.1 * interpolated_spacing ** 2, 8)

    u_initial = np.cos(theta)
    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111, projection='3d')
    # cbar = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=u_initial)

    # ===== numerical simulation ====
    forward_folder = '../data/case2_sphere/forward/'

    clip_bool = False  # True for 2D grid to ensure interpolated coord does not exist domain. False for sphere and heart
    bc_type = 'neumann'

    DT = dt.copy()
    DURATION = duration
    INTERPOLATED_SPACING = interpolated_spacing.copy()
    ORDER_ACC = 4  # central difference by taking two elements right and left each to compute gradient
    ORDER_DERIVATIVE = 2  # diffusion term is second derivative

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    ut = Utils()

    coord = coord
    t = np.zeros([len(coord), ])
    u0 = u_initial.copy()
    D = np.ones([len(coord), ])
    c = np.zeros([len(coord), ])

    # ========================= compute parameters in physics_model =============================== #
    point_cloud = PointCloud()
    point_cloud.assign_coord(coord, t)
    point_cloud.compute_no_pt()
    point_cloud.compute_nn_indices_neighbor(n_neighbors=15, algorithm='kd_tree')
    point_cloud.compute_nn_coord()
    point_cloud.compute_local_axis()
    point_cloud.interpolate_coord_local_axis1_axis2(interpolated_spacing=INTERPOLATED_SPACING, order_acc=ORDER_ACC, \
                                                    order_derivative=ORDER_DERIVATIVE, clip=clip_bool)
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

    # == Boundary Condition only apply for 2D cases ==
    # get the border of 2D grid]
    if bc_type == 'neumann':  # only apply for 2D case
        bc_region_2D = [0.01, 0.99 * point_cloud.coord[:, 0].max(), 0.01, 0.99 * point_cloud.coord[:, 1].max()]
        bc = BoundaryCondition(point_cloud, bc_region_2D)
    if bc_type == 'periodic':  # only apply for close surface
        bc = BoundaryCondition(point_cloud, bc_region_2D=None)
    bc.set_bc_type('neumann')

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
    physics_model.assign_boundary_condition(bc)

    # ============================ numpy solver ===================================================== #
    import time
    start = time.time()
    solver = ForwardSolver(point_cloud, physics_model, interpolated_spacing=INTERPOLATED_SPACING, order_acc=ORDER_ACC)
    solver.generate_first_der_coeff_matrix()
    solver.generate_second_der_coeff_matrix()
    u_update, time_pt = solver.solve(dt=DT, duration=DURATION)
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
    physics_model.assign_t(time_pt)
    physics_model.compute_nn_u()
    physics_model_instances = physics_model.instance_to_dict()

    point_cloud_instances = point_cloud.instance_to_dict()

    # ================ WRITE DOWN THE PARAMETER USED AND U_UPDATE INTO SAV ===================== #
    i = fileio.file_number_forward_README()
    fileio.write_physics_model_instance(physics_model_instances, i, model='diffusion_liang')
    fileio.write_point_cloud_instance(point_cloud_instances, i)

    with open('{}/{}{}.txt'.format(forward_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('dt={}\n'.format(DT))
        csv_file.write('simulation_duration={}\n'.format(DURATION))
        csv_file.write('interpolated_spacing={}\n'.format(INTERPOLATED_SPACING))
        csv_file.write('order_acc={}'.format(ORDER_ACC))
    print('writing {}/{}{}.txt'.format(forward_folder, 'README', i))
