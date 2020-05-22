import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.variables.Variables import Variables
from src.variables.FHNeq     import FHNeq
from src.pointcloud.PointCloud import PointCloud
from src.utils.RbfInterpolator import RbfInterpolatorSphCubic
from src.variables.FiniteDiff import FiniteDiff2ndOrder
from src.utils.DataFrame import DataFrame
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data = pd.read_csv('input.txt', header=None).values
    data_file_name = data[0, 1]
    min_spacing = float(data[1, 1])
    duration = float(data[2, 1])

    dataframe = DataFrame(data_file_name)
    coord = dataframe.get_coord()
    u0 = dataframe.get_uni_u()
    w0 = dataframe.get_uni_w()

    pt_cld = PointCloud(coord, grid_length=0.2, local_grid_resolution=19)
    interp = RbfInterpolatorSphCubic(pt_cld)

    u = Variables(pt_cld, interp, 0)
    w = Variables(pt_cld, interp, 0)
    u.set_val(u0)  # set to some initial values
    w.set_val(w0)  # set to some initial values

    # set up the variables
    D = Variables(pt_cld, interp, 0)
    D.set_val(dataframe.get_D())

    a = Variables(pt_cld, interp, 0)
    b = Variables(pt_cld, interp, 0)
    epsilon_beta = Variables(pt_cld, interp, 0)
    neg_epsilon_gamma = Variables(pt_cld, interp, 0)
    neg_epsilon_delta = Variables(pt_cld, interp, 0)

    a.set_val(dataframe.get_a())
    b.set_val(np.ones(u0.shape))
    epsilon_beta.set_val(dataframe.get_beta())
    neg_epsilon_gamma.set_val(dataframe.get_gamma())
    neg_epsilon_delta.set_val(dataframe.get_delta())

    fhn = FHNeq(a, b,epsilon_beta, neg_epsilon_gamma, neg_epsilon_delta, D, pt_cld, interp, dt=0.01)
    print('interpolated_spacing: ', pt_cld.interpolated_spacing)
    print('dt: ', fhn.dt)

    nstep = int(duration/fhn.dt)
    fhn.integrate(u, w, nstep)

    print('u ', u.val)
    print('w ', w.val)



