import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

import numpy as np
from FileIO import FileIO
from DiffusionDLModel import DiffusionDLModel
import tensorflow as tf
from FiniteDifferenceKernel import FiniteDifferenceKernel
from InverseSolver import InverseSolver
# from ParameterNetwork1 import ParameterNetwork1

if __name__=='__main__':

    forward_folder = '../data/case2_sphere_D1_c0/forward/'
    inverse_folder ='../data/case2_sphere_D1_c0/inverse/'
    i = 1

    BATCH_SIZE = 2000  #0.1/5e-5
    NUM_EPOCH = 5000
    TF_SEED = 4
    LEARNING_RATE = 0.01
    LOSS = tf.losses.MeanSquaredError()
    OPTIMIZER = tf.optimizers.Adam(lr=LEARNING_RATE)


    # ========================= get parameter from README txt file =============================== #
    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.assign_inverse_folder(inverse_folder)
    physics_model_instances = fileio.read_physics_model_instance(i, 'diffusion')
    t = physics_model_instances['t']
    u = physics_model_instances['u']
    nn_u = physics_model_instances['nn_u']

    point_cloud_instances = fileio.read_point_cloud_instance(i)
    nn_indices = point_cloud_instances['nn_indices']
    dist_intp_coord_axis1 = point_cloud_instances['dist_intp_coord_axis1']
    dist_intp_coord_axis2 = point_cloud_instances['dist_intp_coord_axis2']

    dictionary = fileio.read_forward_README_txt(i)
    DT = np.array(dictionary.get('dt'), dtype='float64')
    DURATION = np.array(dictionary.get('simulation_duration'), dtype='float64')
    INTERPOLATED_SPACING = float(dictionary.get('interpolated_spacing'))
    ORDER_ACC = int(dictionary.get('order_acc'))

    # =================== Diffusion DL Model ============
    diffusion_dl_model = DiffusionDLModel()
    diffusion_dl_model.assign_nn_indices(nn_indices)
    diffusion_dl_model.assign_interpolated_spacing(INTERPOLATED_SPACING)
    diffusion_dl_model.assign_dist_intp_coord_axis1(dist_intp_coord_axis1)
    diffusion_dl_model.assign_dist_intp_coord_axis2(dist_intp_coord_axis2)
    diffusion_dl_model.assign_t(t)
    diffusion_dl_model.assign_u(u)
    diffusion_dl_model.compute_no_pt()
    # diffusion_dl_model.compute_nn_u()

    intp_u_axis1, intp_u_axis2 = diffusion_dl_model.interpolate_u_axis1_axis2(ORDER_ACC)
    diffusion_dl_model.assign_intp_u_axis1(intp_u_axis1)
    diffusion_dl_model.assign_intp_u_axis2(intp_u_axis2)

    diffusion_dl_model.initialize_weight()
    dudt = diffusion_dl_model.compute_dudt()
    diffusion_dl_model.assign_dudt(dudt)

    # ============================================================================================ #
    inverse_solver = InverseSolver(diffusion_dl_model, ORDER_ACC)
    coeff_matrix_first_der = inverse_solver.generate_first_der_coeff_matrix()
    coeff_matrix_second_der = inverse_solver.generate_second_der_coeff_matrix()
    inverse_solver.convert_coeff_matrix_to_tensor_with_correct_shape(coeff_matrix_first_der, coeff_matrix_second_der)
    n_training_epoch, training_loss = inverse_solver.solve(NUM_EPOCH, BATCH_SIZE, LOSS, OPTIMIZER)

    # ======================================================
    diffusion_dl_model.assign_training_loss(n_training_epoch, training_loss)
    diffusion_dl_model_instance = diffusion_dl_model.instance_to_dic()

    # ================ WRITE DOWN THE PARAMETER USED AND WEIGHTS INTO SAV ===================== #
    i = fileio.file_number_inverse_README()
    fileio.write_inverse_physics_model_instance(diffusion_dl_model_instance, i, model='diffusion')

    with open('{}/{}{}.txt'.format(inverse_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('batch_size={}\n'.format(BATCH_SIZE))
        csv_file.write('num_epoch={}\n'.format(NUM_EPOCH))
        csv_file.write('tf_seed={}\n'.format(TF_SEED))
        csv_file.write('learning_rate={}'.format(LEARNING_RATE))
        csv_file.write('loss_method={}\n'.format(LOSS.name))
        csv_file.write('optimizer={}'.format(OPTIMIZER.get_config()['name']))

    print('writing {}/{}{}.txt'.format(inverse_folder, 'README', i))
