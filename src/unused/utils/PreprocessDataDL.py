import sys
sys.path.insert(1, '/home/sawsn/Shiernee/TachySolver2/src/utils')

import numpy as np


class PreprocessDataDL:
    def __init__(self, physics_model, README_txt):

        self.physics_model = physics_model

        # ========================= get parameter from README txt file =============================== #
        case0_folder = FileIO(folder=data_path + folder)
        data = case0_folder.read_README_txt(README_txt)
        self.DT = np.array(data.get('dt'), dtype='float64')
        self.DURATION = np.array(data.get('simulation_duration'), dtype='float64')
        self.INTERPOLATED_SPACING = np.array(data.get('interpolated_spacing'), dtype='float64')
        self.ORDER_ACC = np.array(data.get('order_acc'), dtype='int')

        self.model_coord = PhysicsModelCoord(case0_folder, u_file=u_file_sav, u_file_type='sav')
        self.model_DC = PhysicsModelDC(case0_folder, self.model_coord)
        self.dudt_true = self.get_dudt_true()
        self.N_PT = self.model_coord.coord.shape[0]


    def get_dudt_true(self):
        # ========== Prepare x, y train input and parameters for network ======== #
        dt = self.physics_model.t[2] - self.physics_model.t[1]
        u = self.physics_model.u.copy()
        dudt_true = []

        for n in range(len(u) - 1):
            tmp = np.array(u[n + 1], dtype='float64') - np.array(u[n], dtype='float64')
            dudt_true.append(tmp / dt)
        dudt_true = np.array(dudt_true, dtype='float64')
        print('Parameters needed for Deep learning - DONE ')
        # =========================================================
        return dudt_true

    def prepare_input_shape(self):
        #TODO: have to check this again. what shape is this.
        intp_u_axis1, intp_u_axis2 = self.physics_model.intp_u_axis1.copy(), self.physics_model.intp_u_axis2.copy()

        tmp = np.concatenate((intp_u_axis1, intp_u_axis2), axis=-1)
        np.testing.assert_array_equal(tmp[:, :, 2], tmp[:, :, 7]), 'intp_u_axis1 and intp_u_axis2 do not have ' \
                                                                   'the same int u after concatenation'
        input_to_DL = np.delete(tmp, 7, axis=-1)
        return input_to_DL
