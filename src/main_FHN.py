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

    # TODO change min_spacing to grid size
    pt_cld = PointCloud(coord, min_spacing)
    interp = RbfInterpolatorSphCubic(pt_cld)
    finite_diff = FiniteDiff2ndOrder()

    u = Variables(pt_cld, interp, 0)
    w = Variables(pt_cld, interp, 0)
    u.set_val(u0)  # set to some initial values
    w.set_val(w0)  # set to some initial values

    # set up the variables
    D = Variables(pt_cld, interp, 0)
    D.set_val(dataframe.get_D())

    a = np.ones(u0.shape)
    b = np.ones(u0.shape)
    c = np.ones(u0.shape)
    d = np.ones(u0.shape)

    # a = Variables(pt_cld, interp, 0)
    # epsilon = Variables(pt_cld, interp, 0)
    # beta = Variables(pt_cld, interp, 0)
    # gamma = Variables(pt_cld, interp, 0)
    # delta = Variables(pt_cld, interp, 0)
    #
    # a.set_val(dataframe.get_a())
    # epsilon.set_val(dataframe.get_epsilon())
    # beta.set_val(dataframe.get_beta())
    # delta.set_val(dataframe.get_delta())

    fhn = FHNeq(a, b, c, d, D, pt_cld, interp, finite_diff)
    print('interpolated_spacing: ', pt_cld.interpolated_spacing)
    print('dt: ', fhn.dt)

    nstep = int(duration/fhn.dt)
    fhn.integrate(u, w, nstep)

    print('u ', u.val)
    print('w ', w.val)



