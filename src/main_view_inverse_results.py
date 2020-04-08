import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case2_sphere_D(diff)_c0/forward/'
    inverse_folder = '../data/case2_sphere_D(diff)_c0/inverse/'

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.assign_inverse_folder(inverse_folder)
    i_forward = 1
    i_inverse = 1
    physics_model_instances = fileio.read_physics_model_instance(i_forward, model='diffusion')
    physics_dl_model_instances = fileio.read_inverse_physics_model_instance(i_inverse, model='diffusion')
    point_cloud_instances = fileio.read_point_cloud_instance(i_forward)

    coord = point_cloud_instances['coord']
    t = physics_model_instances['t']
    u = physics_model_instances['u']
    D = physics_model_instances['D']
    c = physics_model_instances['c']
    tf_weight_D = physics_dl_model_instances['tf_lowest_weight_D']
    weight_D = np.squeeze(tf_weight_D.numpy())
    # tf_weight_c = physics_dl_model_instances['tf_lowest_weight_c']
    # weight_c = np.squeeze(tf_weight_c.numpy())
    training_loss = physics_dl_model_instances['training_loss']
    n_training_epoch = physics_dl_model_instances['n_training_epoch']

    print('mean: {}, range: {} - {}'.format(weight_D.mean(), weight_D.min(), weight_D.max()))
    print('error(%): {} \u00B1 {}'.format((abs(weight_D - D)/D).mean()*100, (abs(weight_D - D)/D).std()*100))
    print('quartile error(%): {} \u00B1 {}'.format(np.quantile((abs(weight_D - D) / D), 0.25) * 100,
                                                   np.quantile((abs(weight_D - D) / D), 0.75) * 100))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_training_epoch, training_loss)
    ax.set_xlabel('n_training_epoch')
    ax.set_ylabel('training_loss')
    ax.set_title('training_loss')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(weight_D, '.', label='predicted D')
    ax2.plot(D, '.',  label='true D')
    ax2.set_xlabel('point_id')
    ax2.set_ylabel('Value of D')
    ax2.set_title('Value of D')
    ax2.legend()

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.plot((weight_D - D)/weight_D*100, '.')
    ax3.set_xlabel('point_id')
    ax3.set_ylabel('% error of D')
    ax3.set_title('% error of D')
    ax3.legend()

    fig.tight_layout()

    # plt.show()

    fileio.save_inverse_result_file(i_inverse, model='diffusion', fig_name='loss_predicted_D')
