"""
This class is to read files in a specific folder
"""

import numpy as np
import csv
import os
import pickle
from sklearn.externals import joblib
import GlobalParameters as gp
# impo/rt Hyperparameters as hp
import matplotlib.pyplot as plt

class FileIO23:
    def __init__(self, folder=None):
        self.folder = folder

    def read_u_file_csv(self, u_file_csv):
        print('Reading {}/{}.csv'.format(self.folder, u_file_csv))
        reader = csv.reader(open('{}/{}.csv'.format(self.folder, u_file_csv)))
        data = np.array(list(reader)[1:], dtype='float64')
        coord, t, u = data[:, :3], data[:, 3], data[:, 4]

        if gp.DEBUG is True:
            self.check_read_u_file(coord, t, u)

        return coord, t, u

    def read_DC_file_csv(self, DC_file_csv):
        print('Reading {}/{}.csv'.format(self.folder, DC_file_csv))
        reader = csv.reader(open('{}/{}.csv'.format(self.folder, DC_file_csv)))
        data = np.array(list(reader)[1:], dtype='float64')
        coord, D, c = data[:, :3], data[:, 3], data[:, 4]

        if gp.DEBUG is True:
            self.check_read_DC_file(coord, D, c)

        return coord, D, c

    def read_predicted_DC_file_sav(self, DC_file_sav):
        print('Reading {}/{}.sav'.format(self.folder, DC_file_sav))
        data = joblib.load(open('{}/{}.sav'.format(self.folder, DC_file_sav), "rb"))
        coord, D, c = data[:, :3], data[:, 3], data[:, 4]

        if gp.DEBUG is True:
            self.check_read_DC_file(coord, D, c)

        return coord, D, c

    def read_loss_file_csv(self, i):
        print('Reading {}/loss{}.csv'.format(self.folder, i))
        reader = csv.reader(open('{}/loss{}.csv'.format(self.folder, i)))
        data = np.array(list(reader)[1:], dtype='float64')
        Niter, loss = data[:, 0], data[:, 1]

        return Niter, loss

    def read_loss_file_sav(self, i):
        print('Reading {}/loss{}.sav'.format(self.folder, i))
        data = joblib.load(open('{}/loss{}.sav'.format(self.folder, i), "rb"))
        Niter, loss = data[:, 0], data[:, 1]

        return Niter, loss

    def read_README_txt(self, README_txt):
        print('Reading {}/{}.txt'.format(self.folder, README_txt))
        d = {}
        with open('{}/{}.txt'.format(self.folder, README_txt), 'r') as f:
            for line in f:
                (key, val) = line.split('=')
                d[key] = val
        return d

    def read_u_file_pkl(self, u_file_pkl):
        print('Reading {}/{}.pickle'.format(self.folder, u_file_pkl))
        data = pickle.load(open('{}/{}.pickle'.format(self.folder, u_file_pkl), "rb"))
        coord, t, u = data[:, :3], data[:, 3], data[:, 4]

        if gp.DEBUG is True:
            self.check_u_file(coord, t, u=None)

        return coord, t, u

    def read_u_file_sav(self, u_file_sav):
        print('Reading {}/{}.pickle'.format(self.folder, u_file_sav))
        data = joblib.load(open('{}/{}.sav'.format(self.folder, u_file_sav), "rb"))
        coord, t, u = data[:, :3], data[:, 3], data[:, 4]
        coord = coord[t == 0]

        if len(t) == 1:
            dt = None
        else:
            t = np.unique(t)
        return coord, t, u

    def read_ind_sampled_pt_csv(self, index_sampled_pt_csv):
        print('Reading {}/{}.csv'.format(self.folder, index_sampled_pt_csv))
        reader = csv.reader(open('{}/{}.csv'.format(self.folder, index_sampled_pt_csv)))
        ind = np.array(list(reader)[1:], dtype='float64')
        return ind

    def write_u_file_csv(self, coord, t, u, type):
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert np.ndim(t) == 1, 'time shape should be an (time_pt, )'
        assert np.ndim(u) == 2, 'u shape should be an (time_pt x n_coord)'

        if len(t) == 1:
            coord_towrite = coord
        else:
            dt = np.diff(t)
            assert np.sum(dt - dt[0]) < 1e-10, 'dt is not consistent'
            repeat_no = int(t[-1] / dt[0]) + 1
            coord_towrite = np.tile(coord, (repeat_no, 1))
            t_towrite = np.repeat(t, coord.shape[0])

        data_to_write = np.zeros([coord_towrite.shape[0], 5])
        data_to_write[:, :3] = coord_towrite
        data_to_write[:, 3] = t_towrite
        data_to_write[:, 4] = u.flatten()

        # check if file exists or not
        i = self.file_number('u_file_from_{}_DC_test'.format(type))
        filename = '{}/u_file_from_{}_DC_test{}.csv'.format(self.folder, type, i)
        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['x', 'y', 'z', 't', 'u'])
            for value in data_to_write:
                writer.writerow(value)
        print('finish writing {}'.format(filename))
        return

    def write_u_file_pkl(self, coord, t, u, i, type):
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert np.ndim(t) == 1, 'time shape should be an (time_pt, )'
        assert np.ndim(u) == 2, 'u shape should be an (time_pt x n_coord)'

        if len(t) == 1:
            coord_towrite = coord
        else:
            dt = np.diff(t)
            assert np.sum(dt - dt[0]) < 1e-10, 'dt is not consistent'
            repeat_no = int(t[-1]/dt[0]) + 1
            coord_towrite = np.tile(coord, (repeat_no, 1))
            t_towrite = np.repeat(t, coord.shape[0])

        data_to_write = np.zeros([coord_towrite.shape[0], 5])
        data_to_write[:, :3] = coord_towrite
        data_to_write[:, 3] = t_towrite
        data_to_write[:, 4] = u.flatten()

        # check if file exists or not
        filename = '{}/u_file_from_{}_DC_test{}.pickle'.format(self.folder, type, i)
        if os.path.exists(filename) is True:
            raise Exception('{} already exist.'.format(filename))

        pickle_out = open(filename, "wb")
        pickle.dump(data_to_write, pickle_out)
        pickle_out.close()
        print('finish writing {}'.format(filename))
        return

    def write_u_file_sav(self, coord, t, u, i, type):
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert np.ndim(t) == 1, 'time shape should be an (time_pt, )'
        assert np.ndim(u) == 2, 'u shape should be an (time_pt x n_coord)'

        if len(t) == 1:
            coord_towrite = coord
            t_towrite = t
        else:
            dt = np.diff(t)
            assert np.sum(dt - dt[0]) < 1e-10, 'dt is not consistent'
            repeat_no = int(t[-1]/dt[0]) + 1
            coord_towrite = np.tile(coord, (repeat_no, 1))
            t_towrite = np.repeat(t, coord.shape[0])

        data_to_write = np.zeros([coord_towrite.shape[0], 5])
        data_to_write[:, :3] = coord_towrite
        data_to_write[:, 3] = t_towrite
        data_to_write[:, 4] = u.flatten()

        # check if file exists or not
        filename = '{}/u_file_from_{}_DC_test{}.sav'.format(self.folder, type, i)
        if os.path.exists(filename) is True:
            raise Exception('{} already exist.'.format(filename))

        joblib.dump(data_to_write, filename)
        print('finish writing {}'.format(filename))
        return

    # def overwrite_u_file_sav(self, coord, t, u, i, type):
    #     assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
    #     assert np.ndim(t) == 1, 'time shape should be an (time_pt, )'
    #     assert np.ndim(u) == 2, 'u shape should be an (time_pt x n_coord)'
    #
    #     if len(t) == 1:
    #         coord_towrite = coord
    #     else:
    #         dt = np.diff(t)
    #         assert np.sum(dt - dt[0]) < 1e-10, 'dt is not consistent'
    #         repeat_no = int(t[-1]/dt[0]) + 1
    #         coord_towrite = np.tile(coord, (repeat_no, 1))
    #         t_towrite = np.repeat(t, coord.shape[0])
    #
    #     data_to_write = np.zeros([coord_towrite.shape[0], 5])
    #     data_to_write[:, :3] = coord_towrite
    #     data_to_write[:, 3] = t_towrite
    #     data_to_write[:, 4] = u.flatten()
    #
    #     filename = '{}/u_file_from_{}_DC_test{}.sav'.format(self.folder, type, i)
    #
    #     joblib.dump(data_to_write, filename)
    #     print('finish writing {}'.format(filename))
    #     return

    def write_predicted_DC_file_csv(self, coord, D, c, i):
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert np.ndim(D) == 1, 'D shape should be an (n_coord, )'
        assert np.ndim(c) == 1, 'c shape should be an (n_coord, )'

        data = np.concatenate((coord, D.reshape([-1, 1])), axis=1)
        data = np.concatenate((data, c.reshape([-1, 1])), axis=1)

        filename = '{}/predicted_DC{}.csv'.format(self.folder, i)

        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['x', 'y', 'z', 'D', 'c'])
            for value in data:
                writer.writerow(value)
        print('finish writing {}'.format(filename))
        return

    def write_predicted_DC_file_sav(self, coord, D, c, i):
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert np.ndim(D) == 1, 'D shape should be an (n_coord, )'
        assert np.ndim(c) == 1, 'c shape should be an (n_coord, )'

        repetition = int(len(D) / len(coord))
        coord = np.tile(coord, [repetition, 1])
        data = np.concatenate((coord, D.reshape([-1, 1])), axis=1)
        data = np.concatenate((data, c.reshape([-1, 1])), axis=1)

        filename = '{}/predicted_DC{}.sav'.format(self.folder, i)
        joblib.dump(data, filename)
        print('finish writing {}'.format(filename))
        return

    def write_loss_file_csv(self, Niter, loss):
        assert np.ndim(Niter) == 1, 'Niter shape should be an (n_coord, )'
        assert np.ndim(loss) == 1, 'loss shape should be an (n_coord, )'

        i = self.file_number('loss')
        data = np.concatenate((Niter.reshape([-1, 1]), loss.reshape([-1, 1])), axis=1)

        filename = '{}/loss{}.csv'.format(self.folder, i)
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['Niter', 'loss'])
            for value in data:
                writer.writerow(value)
        print('finish writing {}'.format(filename))
        return i

    def write_dudt_label_file_csv(self, coord, dudt_label, i):
        assert coord.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert np.ndim(dudt_label) == 2, 'loss shape should be an (time_pt, n_coord)'

        data = np.concatenate((coord, np.transpose(dudt_label)), axis=1)

        filename = '{}/dudt_label_{}.csv'.format(self.folder, i)
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['x', 'y', 'z', 'dudt'])
            for value in data:
                writer.writerow(value)
        print('finish writing {}'.format(filename))
        return

    def write_ind_sampled_pt(self, ind, i):
        assert np.ndim(ind) == 1
        ind = ind.reshape([len(ind), 1])

        filename = '{}/index_sampled_pt_{}.csv'.format(self.folder, i)
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['index of sampled_pt'])
            for value in ind:
                writer.writerow(value)
        print('finish writing {}'.format(filename))
        return

    def write_loss_file_sav(self, Niter, loss, i):
        assert np.ndim(Niter) == 1, 'Niter shape should be an (n_coord, )'
        assert np.ndim(loss) == 1, 'loss shape should be an (n_coord, )'

        data = np.concatenate((Niter.reshape([-1, 1]), loss.reshape([-1, 1])), axis=1)

        filename = '{}/loss{}.sav'.format(self.folder, i)
        joblib.dump(data, filename)
        print('finish writing {}'.format(filename))
        return i

    def check_read_u_file(self, coord, t, u):
        if np.ndim(coord) != 2:
            raise Exception('coordinates should be 2D array')
        if coord.shape[-1] != 3:
            raise Exception('coordinate last dimension is not 3')
        if np.ndim(t) != 1:
            raise Exception('time should be 1D array')
        if np.min(t) < 0.0:
            raise Exception('negative t is found')
        if np.ndim(u) != 1:
            raise Exception('u should be 1D array')
        print('read_u_file_csv function PASS ')
        return

    def check_read_DC_file(self, coord, D, c):
        if np.ndim(coord) != 2:
            raise Exception('coordinates should be 2D array')
        if np.ndim(D) != 1:
            raise Exception('D should be 1D array')
        if np.ndim(c) != 1:
            raise Exception('c should be 1D array')
        print('read_DC_file_csv function PASS ')
        return

    def check_read_loss_file(self, Niter, loss):
        if np.min(Niter) < 0.0:
            raise Exception('Negative Niter is found')
        if np.min(loss) < 0.0:
            raise Exception('Negative loss is found')
        print('read_loss_file_csv function PASS ')
        return

    def write_parameter_used_for_deep_learning_model(self, i):
        with open('{}/{}_{}.txt'.format(self.folder, hp.README_TXT, i), mode='w+', newline='') as csv_file:
            csv_file.write('batch_size={}\n'.format(hp.BATCH_SIZE))
            csv_file.write('num_epochs={}\n'.format(hp.NUM_EPOCHS))
            csv_file.write('tf_seed={}\n'.format(hp.TF_SEED))
            csv_file.write('learning_rate={}\n'.format(hp.LEARNING_RATE))
            csv_file.write('loss={}\n'.format(hp.LOSS))
            csv_file.write('optimizer={}'.format(hp.OPTIMIZER))
        print('writing {}/{}_{}.txt'.format(self.folder, hp.README_TXT, i))
        return

    def write_physics_model_instance(self, physics_model_instances, i):
        filename = '{}/physics_model_instances{}.sav'.format(self.folder, i)
        joblib.dump(physics_model_instances, filename)
        print('finish writing {}'.format(filename))
        return

    def write_point_cloud_instance(self, point_cloud_instances, i):
        filename = '{}/point_cloud_instances{}.sav'.format(self.folder, i)
        joblib.dump(point_cloud_instances, filename)
        print('finish writing {}'.format(filename))
        return

    def write_forward_README(self, dt, duration, interpolated_spacing, order_acc, i):
        with open('{}/{}{}.txt'.format(self.folder, 'README', i), mode='w', newline='') as csv_file:
            csv_file.write('dt={}\n'.format(dt))
            csv_file.write('simulation_duration={}\n'.format(duration))
            csv_file.write('interpolated_spacing={}\n'.format(interpolated_spacing))
            csv_file.write('order_acc={}'.format(order_acc))
        print('writing {}/{}{}.txt'.format(self.folder, 'README', i))
        return

    def read_physics_model_instance(self, i):
        filename = '{}/physics_model_instances{}.sav'.format(self.folder, i)
        physics_model_instances = joblib.load(filename)
        return physics_model_instances

    def read_point_cloud_instance(self, i):
        filename = '{}/point_cloud_instances{}.sav'.format(self.folder, i)
        point_cloud_instances = joblib.load(filename)
        return point_cloud_instances

    def save_u_theta_png_file(self, i):
        filename = '{}/physics_model_instances{}_u_theta.png'.format(self.folder, i)
        plt.savefig(filename)