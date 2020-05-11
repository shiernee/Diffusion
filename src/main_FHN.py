import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.variables.Variables import Variables
from src.variables.FHNeq     import FHNeq
from src.pointcloud.PointCloud import PointCloud
from src.utils.RbfInterpolator import RbfInterpolatorSph
from src.variables.FiniteDiff2ndOrder import FiniteDiff2ndOrder
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
    D = dataframe.get_D()

    pt_cld = PointCloud(coord, min_spacing)
    interp = RbfInterpolatorSph(pt_cld)
    finite_diff = FiniteDiff2ndOrder()

    u = Variables(pt_cld, interp, finite_diff, 0)
    w = Variables(pt_cld, interp, finite_diff, 0)
    u.set_val(u0)  # set to some initial values
    w.set_val(w0)  # set to some initial values
    
    u.eval_ddx()
    dudx = u.get_ddx() # this is grad_variables type

    # set up the variables
    D = D
    a = np.ones(u.val.shape)
    b = np.ones(u.val.shape)
    c = np.ones(u.val.shape)
    d = np.ones(u.val.shape)

    fhn = FHNeq(a, b, c, d, D, pt_cld, interp, finite_diff)
    print('interpolated_spacing: ', pt_cld.interpolated_spacing)
    print('dt: ', fhn.dt)

    nstep = int(duration/fhn.dt)
    fhn.integrate(u, w, nstep)

    print('u ', u.val)
    print('w ', w.val)



