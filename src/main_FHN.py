import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.variables.Variables import Variables
from src.variables.FHNeq     import FHNeq
from src.pointcloud.PointCloud import PointCloud
from src.utils.RbfInterpolator import RbfInterpolator
from src.utils.Parameters import Parameters
from src.variables.FiniteDiff2ndOrder import FiniteDiff2ndOrder
from src.utils.DataFrame import DataFrame
import numpy as np


if __name__ == '__main__':

    parent_file = path.dirname(path.dirname(path.abspath(__file__)))
    filename = os.path.join(parent_file, "data", "testcase", "database.csv")
    param_file = os.path.join(parent_file, "data", "testcase", "param_template.csv")

    dataframe = DataFrame(filename)
    coord = dataframe.get_coord()
    u0 = dataframe.get_uni_u()
    w0 = dataframe.get_uni_w()
    D = dataframe.get_D()

    pt_cld = PointCloud(coord, param_file)
    print('pt_cloud.coord shape: {}'.format(pt_cld.coord.shape))
    print('pt_cloud.no_pt: {}'.format(pt_cld.no_pt))
    print('pt_cloud.dist_nn shape: {}'.format(pt_cld.dist_nn.shape))
    print('pt_cloud.nn_indices shape: {}'.format(pt_cld.nn_indices.shape))
    print('pt_cloud.local_axis1 shape: {}'.format(pt_cld.local_axis1.shape))
    print('pt_cloud.local_axis2 shape: {}'.format(pt_cld.local_axis2.shape))
    print('pt_cloud.interpolated_spacing: {}'.format(pt_cld.interpolated_spacing))
    print('pt_cloud.local_grid shape: {}'.format(pt_cld.local_grid.shape))
    print('==============================================================')

    param = Parameters(param_file, D)
    print('param.duration : {}'.format(param.duration))
    print('param.dt : {}'.format(param.dt))
    print('param.nstep : {}'.format(param.nstep))
    print('==============================================================')

    interp = RbfInterpolator(pt_cld)
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

    fhn = FHNeq(a, b, c, d, D, param.dt, pt_cld, interp, finite_diff)

    fhn.integrate(u, w, param.nstep)
    print('u ', u.val)
    print('w ', w.val)



