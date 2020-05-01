import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import tensorflow as tf
from Utils import Utils


class InverseSolver:
    def __init__(self, physics_dl_model, order_acc):
        self.physics_dl_model = physics_dl_model

        self.order_acc = order_acc
        self.ut = Utils()

        self.tf_coeff_matrix_first_der = None
        self.tf_coeff_matrix_second_der = None

        return

    def solve(self, num_epochs, batch_size, tf_loss, tf_optimizer):
        duration = self.physics_dl_model.t[-1]
        dt = self.physics_dl_model.t[1] - self.physics_dl_model.t[0]

        tf_intp_u_axis1 = self.physics_dl_model.tf_intp_u_axis1.__copy__()
        tf_intp_u_axis2 = self.physics_dl_model.tf_intp_u_axis2.__copy__()
        tf_coeff_matrix_first_der = self.tf_coeff_matrix_first_der.__copy__()
        tf_coeff_matrix_second_der = self.tf_coeff_matrix_second_der.__copy__()
        tf_dudt = self.physics_dl_model.tf_dudt.__copy__()

        n_training_epoch = []
        training_loss = []
        batch_iteration = int(int(duration / dt) / batch_size)

        lowest_loss_value = 1e9
        for epoch in range(num_epochs):
            loss_value = 0

            for iteration in range(batch_iteration):
                begin = [iteration * batch_size, 0, 0]
                size = [batch_size, tf_intp_u_axis1.shape[1], tf_intp_u_axis1.shape[2]]
                tf_intp_u_axis1_tmp = tf.slice(tf_intp_u_axis1, begin, size)
                tf_intp_u_axis2_tmp = tf.slice(tf_intp_u_axis2, begin, size)

                begin = [iteration * batch_size, 0, 0]
                size = [batch_size, tf_dudt.shape[1], tf_dudt.shape[2]]
                tf_dudt_tmp = tf.slice(tf_dudt, begin, size)

                with tf.GradientTape(persistent=True) as tape:
                    tf_dudt_pred = self.physics_dl_model.call_dl_model(tf_intp_u_axis1_tmp, tf_intp_u_axis2_tmp,
                                                                       tf_coeff_matrix_first_der,
                                                                       tf_coeff_matrix_second_der, self.order_acc)
                    loss_value += tf_loss(tf_dudt_tmp, tf_dudt_pred)

                # == calculate gradient and update variables =====
                grads = tape.gradient(loss_value, self.physics_dl_model.trainable_weights)
                tf_optimizer.apply_gradients(zip(grads, self.physics_dl_model.trainable_weights))
                del tape

            average_loss = loss_value.numpy() / batch_iteration
            lowest_loss_value = self.check_if_avg_loss_lower_than_lowest_loss_value(lowest_loss_value,
                                                                                    average_loss)
            n_training_epoch.append(epoch)
            training_loss.append(average_loss)

            # ===== learning rate decay ===
            # initial_lr = tf_optimizer.learning_rate
            # decay = 1e-4
            # tf_optimizer.learning_rate = initial_lr * (1 / (1 + decay * epoch))

            if epoch % 1 == 0:
                print('\n')
                print('epoch{}/{} loss{}'.format(epoch, num_epochs, average_loss))
                print('average_weight_D: {}'.format(tf.reduce_mean(self.physics_dl_model.tf_weight_D).numpy()))
                # print('learning rate used: {}'.format(tf_optimizer.get_config()['learning_rate']))

        n_training_epoch = np.array(n_training_epoch)
        training_loss = np.array(training_loss)

        return n_training_epoch, training_loss

    def check_if_avg_loss_lower_than_lowest_loss_value(self, lowest_loss_value, current_epoch_loss_value):
        print('current_loss  = {} lowest_loss = {}'.format(current_epoch_loss_value,
                                                                     lowest_loss_value))
        if current_epoch_loss_value < lowest_loss_value:
            print('assign_lowest_weight_D')
            self.physics_dl_model.assign_tf_lowest_weight_D(self.physics_dl_model.tf_weight_D)
            lowest_loss_value = current_epoch_loss_value.copy()
        return lowest_loss_value

    def generate_first_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length = np.shape(self.physics_dl_model.tf_intp_u_axis1.numpy())[-1]
        coeff_matrix_first_der = self.ut.coeff_matrix_first_order(input_length, coeff)
        coeff_matrix_first_der = coeff_matrix_first_der.copy()
        return coeff_matrix_first_der

    def generate_second_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length2 = np.shape(self.physics_dl_model.tf_intp_u_axis2.numpy())[-1] - len(coeff) + 1
        coeff_matrix_second_der = self.ut.coeff_matrix_first_order(input_length2, coeff)
        coeff_matrix_second_der = coeff_matrix_second_der.copy()
        return coeff_matrix_second_der

    def convert_coeff_matrix_to_tensor_with_correct_shape(self, coeff_matrix_first_der, coeff_matrix_second_der):
        coeff_matrix_first_der = np.expand_dims(coeff_matrix_first_der, axis=0)  # shape(1, 5, 3)
        coeff_matrix_second_der = np.expand_dims(coeff_matrix_second_der, axis=0)  # shape(1, 3, 1)

        # coeff_matrix_first_der = np.expand_dims(coeff_matrix_first_der, axis=0)  # shape(1, 1, 5, 3)
        # coeff_matrix_second_der = np.expand_dims(coeff_matrix_second_der, axis=0)  # shape(1, 1, 3, 1)

        assert np.ndim(coeff_matrix_first_der) == 3, 'coeff_matrix_first_der must be 3D array (1, 5nn, 3)'
        assert np.ndim(coeff_matrix_first_der) == 3, 'coeff_matrix_first_der must be 3D array (1, 3nn, 1)'

        self.tf_coeff_matrix_first_der = tf.convert_to_tensor(coeff_matrix_first_der, dtype=tf.float64)
        self.tf_coeff_matrix_second_der = tf.convert_to_tensor(coeff_matrix_second_der, dtype=tf.float64)
        return

