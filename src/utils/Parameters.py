import numpy as np
import copy as cp
import pandas as pd


class Parameters:
    def __init__(self, param_file, D):

        self.duration = None
        self.dt = None
        self.nstep = None

        # =============================================

        param = pd.read_csv(param_file)
        self.duration = param['duration'].values[0]

        interpolated_spacing = param['interpolated_spacing_value'].values[0]
        self.dt = self.compute_dt(interpolated_spacing, D)

        self.nstep = int(self.duration // self.dt)

    def compute_dt(self, interpolated_spacing, D):
        D = np.max(D)
        self.dt = 0.4 * interpolated_spacing ** 2 / D
        return cp.copy(self.dt)
