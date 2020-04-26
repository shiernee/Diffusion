from src.variables.Variables import Variables
from src.variables.FHNeq     import FHNeq
from src.pointcloud.PointCloud import PointCloud
from src.utils.RbfInterpolator import RbfInterpolator
from src.utils.Parameters import Parameters

import numpy as np


if __name__ == '__main__':

    case = 'case2'

    pt_cld = PointCloud(case)
    print('pt_cloud.coord shape: {}'.format(pt_cld.coord.shape))
    print('pt_cloud.no_pt: {}'.format(pt_cld.no_pt))
    print('pt_cloud.dist_nn shape: {}'.format(pt_cld.dist_nn.shape))
    print('pt_cloud.nn_indices shape: {}'.format(pt_cld.nn_indices.shape))
    print('pt_cloud.local_axis1 shape: {}'.format(pt_cld.local_axis1.shape))
    print('pt_cloud.local_axis2 shape: {}'.format(pt_cld.local_axis2.shape))
    print('pt_cloud.interpolated_spacing: {}'.format(pt_cld.interpolated_spacing))
    print('pt_cloud.local_grid shape: {}'.format(pt_cld.local_grid.shape))
    print('==============================================================')

    param = Parameters(case)
    print('param.duration : {}'.format(param.duration))
    print('param.u0 shape: {}'.format(param.u0.shape))
    print('param.dt : {}'.format(param.dt))
    print('param.nstep : {}'.format(param.nstep))
    print('==============================================================')

    interp = RbfInterpolator(pt_cld)

    u = Variables(pt_cld, interp, 0)
    w = Variables(pt_cld, interp, 0)
    u.set_val(param.u0)  # set to some initial values
    w.set_val(np.zeros(param.u0.shape))  # set to some initial values
    
    u.eval_ddx()
    dudx = u.get_ddx() # this is grad_variables type

    # set up the variables
    D = np.ones(dudx.val.shape) * param.D
    a = np.ones(u.val.shape)
    b = np.ones(u.val.shape)
    c = np.ones(u.val.shape)
    d = np.ones(u.val.shape)

    fhn = FHNeq(a, b, c, d, D, param.dt, pt_cld, interp)

    fhn.integrate(u, w, param.nstep)
    print('u ', u.val)
    print('w ', w.val)



