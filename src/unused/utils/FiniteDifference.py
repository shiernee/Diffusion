
import numpy as np


class FiniteDifference:
    def __init__(self, order_acc):

        self.order_acc = order_acc
        return

    def fd_coeff(self):
        if self.order_acc == 1:
            coeff = np.array([-1, 1], dtype='float64')
        elif self.order_acc == 2:
            coeff = np.array([-1 / 2, 0, 1 / 2], dtype='float64')
        elif self.order_acc == 4:
            coeff = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12], dtype='float64')
        elif self.order_acc == 6:
            coeff = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60], dtype='float64')
        else:
            print('coefficient for order acc of {} has not be implemented')

        return coeff

    def coeff_matrix_first_order(self, len_input_data):
        '''
        :param len_input_data: input_data.shape[0]
        :return: np.array with shape (input_shape.shape[0],
        '''
        coeff = self.fd_coeff()
        output_shape = [len_input_data, len_input_data - len(coeff) + 1]

        coeff_matrix = np.zeros(output_shape)

        for i in range(output_shape[-1]):
            coeff_matrix[i:i + len(coeff), i] = coeff.copy()

        return coeff_matrix


if __name__ == '__main__':
    fd = FiniteDifference(order_acc=4)
    coeff_matrix = fd.coeff_matrix_first_order(9)
    print(coeff_matrix)
    print('------------------')
    fd = FiniteDifference(order_acc=2)
    coeff_matrix = fd.coeff_matrix_first_order(9)
    print(coeff_matrix)