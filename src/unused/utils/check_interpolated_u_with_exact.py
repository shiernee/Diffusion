''' 
this code is to check the error of interpolated u with exact solution between different method 
- using cartersian coordinate to perform rbf interpolate is bad, spherical coordinate is better. 
- rbf with gaussian function has the lowest max_norm_rel_error (1.0) among all methods (multiquadratic, linear
- cubic is used to analyse at no_pt with highest error 

steps:
1. debug liang_et_al.py until line " point_cloud.interpolate_coord_local_axis1_axis2() 
2. copy paste code below. 
created on 19 Apr 2020 by snsaw
'''

## ==================== START ============================================
## section 1: calculate error using different methods. 

from scipy.interpolate import Rbf
var = u0.copy()
t=0
error_m, error_c, error_l, error_g, error_i = [], [], [], [], []
for i in range(point_cloud.no_pt):
   nn_coord = point_cloud.nn_coord[i]
   r, phi, theta = ut.xyz2sph(nn_coord)
   nn_var = var[point_cloud.nn_indices[i]]

   intp_coord = np.concatenate((point_cloud.intp_coord_axis1[i], point_cloud.intp_coord_axis2[i]),
                             axis=0)
   rintp, phiintp, thetaintp = ut.xyz2sph(intp_coord)

   # calculate error 
   exact_sln = np.exp(-2*t)*np.cos(thetaintp)

   rbfi_s = Rbf(r,phi, theta, nn_var, function='multiquadric')
   intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)
   intp_var_nn_s = np.clip(intp_var_nn_s, round(u0.min(),2), round(u0.max(),2))
   error_m.append(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))
   #print(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))

   rbfi_s = Rbf(r,phi, theta, nn_var, function='linear')
   intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)
   intp_var_nn_s = np.clip(intp_var_nn_s, round(u0.min(),2), round(u0.max(),2))
   error_l.append(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))
   #print(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))   

   rbfi_s = Rbf(r,phi, theta, nn_var, function='cubic')
   intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)
   intp_var_nn_s = np.clip(intp_var_nn_s, round(u0.min(),2), round(u0.max(),2))
   error_c.append(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))
   #print(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))

   rbfi_s = Rbf(r,phi, theta, nn_var, function='gaussian')
   intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)
   intp_var_nn_s = np.clip(intp_var_nn_s, round(u0.min(),2), round(u0.max(),2))
   error_g.append(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))
   #print(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))

   rbfi_s = Rbf(r,phi, theta, nn_var, function='inverse')
   intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)
   intp_var_nn_s = np.clip(intp_var_nn_s, round(u0.min(),2), round(u0.max(),2))
   error_i.append(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))
   #print(np.max(abs((exact_sln - intp_var_nn_s) / exact_sln)))

error_m, error_c, error_l = np.array(error_m), np.array(error_c), np.array(error_l)
error_g, error_i = np.array(error_g), np.array(error_i)
print('multiquadric error: max{:.4f} mean{:.4f} std{:.4f}'.format(error_m.max(), error_m.mean(), error_m.std()))
print('linear error: max{:.4f} mean{:.4f} std{:.4f}'.format(error_l.max(), error_l.mean(), error_l.std()))
print('cubic error: max{:.4f} mean{:.4f} std{:.4f}'.format(error_c.max(), error_c.mean(), error_c.std()))
print('gaussian error: max{:.4f} mean{:.4f} std{:.4f}'.format(error_g.max(), error_g.mean(), error_g.std()))
print('inverse error: max{:.4f} mean{:.4f} std{:.4f}'.format(error_i.max(), error_i.mean(), error_i.std()))
plt.figure()
plt.plot(error_m, '.', label='multiquadric')
plt.plot(error_c, 'x', label='cubic')
plt.plot(error_l, '_', label='linear')
plt.plot(error_g, '+', label='gaussian')
plt.plot(error_i, '*', label='inverse')
plt.legend()

##  section 2: ====== interpolate at no_pt with highest error ======= #
i=error_c.argmax()
print('plotting {} no_pt'.format(i))

nn_coord = point_cloud.nn_coord[i]
r, phi, theta = ut.xyz2sph(nn_coord)
nn_var = var[point_cloud.nn_indices[i]]

# interpolation 
x, y, z = nn_coord[:, 0], nn_coord[:, 1], nn_coord[:, 2]
rbfi = Rbf(x, y, z, nn_var, function='cubic')
rbfi_s = Rbf(r,phi, theta, nn_var, function='cubic')

# =====
intp_coord = np.concatenate((point_cloud.intp_coord_axis1[i], point_cloud.intp_coord_axis2[i]),
                             axis=0)
rintp, phiintp, thetaintp = ut.xyz2sph(intp_coord)
xi, yi, zi = intp_coord[:, 0], intp_coord[:, 1], intp_coord[:, 2]
intp_var_nn = rbfi(xi, yi, zi)
intp_var_nn_s = rbfi_s(rintp, phiintp, thetaintp)
intp_var_nn = np.clip(intp_var_nn, round(u0.min(),2), round(u0.max(),2))
intp_var_nn_s = np.clip(intp_var_nn_s, round(u0.min(),2), round(u0.max(),2))

exact_sln = np.exp(-2*t)*np.cos(thetaintp)

# meshgrid
nn = 30

RI, PI, TI = np.ones([nn, ]), np.linspace(phi.min(), phi.max(), nn), \
             np.linspace(theta.min(), theta.max(), nn)
RI, PI, TI = np.meshgrid(RI, PI, TI)
RI, PI, TI = RI.flatten(), PI.flatten(), TI.flatten()
SPH_COORD = np.array([RI, PI, TI]).T
XI, YI, ZI = ut.sph2xyz(SPH_COORD)

intp_var_mesh = rbfi(XI, YI, ZI)
intp_var_mesh_s = rbfi_s(RI, PI ,TI)
intp_var_mesh = np.clip(intp_var_mesh, round(u0.min(),2), round(u0.max(),2))
intp_var_mesh_s = np.clip(intp_var_mesh_s, round(u0.min(),2), round(u0.max(),2))


# plotting
#bb = np.concatenate((nn_var, intp_var_s), axis=0)
bb = nn_var.copy()

plt.figure()
plt.scatter(PI, TI, c=intp_var_mesh_s, vmin=bb.min(), vmax=bb.max())
plt.scatter(phi, theta, c=nn_var, vmin=bb.min(), vmax=bb.max(), edgecolors='k')
plt.scatter(phiintp, thetaintp, c=intp_var_nn_s, vmin=bb.min(), vmax=bb.max(), edgecolors='r')

for j in range(len(phi)):
   plt.annotate(round(nn_var[j],3), (phi[j], theta[j]))
for j in range(len(phiintp)):
   plt.annotate(round(intp_var_nn_s[j],3), (phiintp[j], thetaintp[j]))
plt.title('cubic spherical interpolation')


''' plot exact sln '''
plt.figure()
plt.scatter(PI, TI, c=intp_var_mesh_s, vmin=bb.min(), vmax=bb.max())
plt.scatter(phi, theta, c=nn_var, vmin=bb.min(), vmax=bb.max(), edgecolors='k')
plt.scatter(phiintp, thetaintp, c=exact_sln, vmin=bb.min(), vmax=bb.max(), edgecolors='r')

for j in range(len(phi)):
   plt.annotate(round(nn_var[j],3), (phi[j], theta[j]))
for j in range(len(phiintp)):
   plt.annotate(round(exact_sln[j],3), (phiintp[j], thetaintp[j]))
plt.title('exact multiquadric spherical interpolation')

''' plot err sln '''
error = abs((exact_sln - intp_var_nn_s) / exact_sln)
plt.figure()
plt.scatter(PI, TI, c=intp_var_mesh_s, vmin=bb.min(), vmax=bb.max())
plt.scatter(phi, theta, c=nn_var, vmin=bb.min(), vmax=bb.max(), edgecolors='k')
plt.scatter(phiintp, thetaintp, c=error, vmin=bb.min(), vmax=bb.max(), edgecolors='r')

for j in range(len(phi)):
   plt.annotate(round(nn_var[j],3), (phi[j], theta[j]))
for j in range(len(phiintp)):
   plt.annotate(round(error[j],3), (phiintp[j], thetaintp[j]))
plt.title('error multiquadric spherical interpolation')

# ============================== end ==================================


''' 
# ====  section 3: cartersian interpolation ========
exact_sln = np.cos(thetaintp)
plt.figure()
plt.scatter(PI, TI, c=intp_var_mesh, vmin=bb.min(), vmax=bb.max())
plt.scatter(phi, theta, c=nn_var, vmin=bb.min(), vmax=bb.max(), edgecolors='k')
plt.scatter(phiintp, thetaintp, c=intp_var_nn, vmin=bb.min(), vmax=bb.max(), edgecolors='r')

for j in range(len(phi)):
   plt.annotate(round(intp_var_mesh[j],3), (phi[j], theta[j]))
for j in range(len(phiintp)):
   plt.annotate(round(intp_var_nn[j],3), (phiintp[j], thetaintp[j]))
plt.title('cartesian interpolation')
'''






