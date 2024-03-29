'''
this code is to check if intp_u_axis1, and intp_u_axis2 generated from
self.interplate_rbf(u) in Diffusion_Model.py is correct

step:
1. debug liang_et_al.py till Diffusion_Model.py line "intp_u_axis1, intp_u_axis2 = self.interpolate_rbf(u)"
2. copy paste code below to check.
3. cubic error max should be around 14

created on 19 Apr 2020 by snsaw
'''

import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from Utils.Utils import xyz2r_phi_theta

var = u.copy()
error_c= []
for i in range(self.point_cloud.no_pt):
    intp_coord = np.concatenate((self.point_cloud.intp_coord_axis1[i],
                                 self.point_cloud.intp_coord_axis2[i]),
                                 axis=0)
    x, y, z = intp_coord[:, 0], intp_coord[:, 1], intp_coord[:, 2]
    rintp, phiintp, thetaintp = xyz2sph(x, y, z)
    exact_sln = np.cos(thetaintp)

    numerical_sln = np.concatenate((intp_u_axis1[i], intp_u_axis2[i]), axis=0)
    error_c.append(np.max(abs((exact_sln - numerical_sln) / exact_sln)))

error_c = np.array(error_c)
print('cubic error: max{:.4f} mean{:.4f} std{:.4f}'.format(error_c.max(), error_c.mean(), error_c.std()))
