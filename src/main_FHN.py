from src.variables.variables import variables
from src.variables.FHNeq     import FHNeq
from src.pointcloud.GeneratePoints import GeneratePoints
from src.pointcloud.PointCloud import PointCloud
from src.utils.Parameters import Parameters
from src.utils.Rbf_interpolator import Rbf_interpolator

import numpy as np

# ================================

if __name__ == '__main__':

    np.random.seed(23873)
    p = Parameters()

    gen_pt = GeneratePoints() # HK - only read points from a file

    # HK
    cart_coord, sph_coord = gen_pt.generate_points_sphere(no_pt=p.no_pt, sph_radius=1)
    theta = sph_coord[:, 2]
    p.set_u_initial(theta=theta)

    pt_cld = PointCloud(cart_coord,p.interpolated_spacing_method,p.neighbours,p.order_acc)
    p.compute_dt(pt_cld.interpolated_spacing)

    interp = Rbf_interpolator(pt_cld)

    u = variables(pt_cld, interp, 0)
    w = variables(pt_cld, interp, 0)
    u.set_val(p.u0)  # set to some initial values
    w.set_val(np.zeros(p.u0.shape))  # set to some initial values
    
    u.eval_ddx()
    dudx = u.get_ddx() # this is grad_variables type

    # set up the variables
    D = np.ones(dudx.val.shape) * p.D
    a = np.ones(u.val.shape)
    b = np.ones(u.val.shape)
    c = np.ones(u.val.shape)
    d = np.ones(u.val.shape)

    fhn = FHNeq(a,b,c,d,D,p.dt,pt_cld,interp)

    nsteps = 10
    fhn.integrate(u,w,nsteps)
    print('u ',u.val)
    print('w ',w.val)



