'''
this code is used to plot the nn_neigbour_intp_coord and intp_u value to check why intp_u value is not correct. 

created on 18 Apr 2020 by snsaw.
'''

aa = self.point_cloud.coord[self.point_cloud.nn_indices_intp_coord_axis1]  #4002, 9, 24, 3
self.point_cloud.intp_coord_axis1 # 4002, 9, 3
self.point_cloud.dist_intp_coord_axis1 #4002, 9, 24
nn_u_axis1  #4002, 9, 24
intp_u_axis1 #4002, 9

import matplotlib.pyplot as plt
aa = self.point_cloud.coord[self.point_cloud.nn_indices_intp_coord_axis1]  #4002, 9, 24, 3
nn_u_axis1 = nn_u_axis1.reshape([-1, 9, 24])
fig = plt.figure()
ax = fig.add_subplot(111)
for nn in range(9):
   r_, phi_, theta_  = self.ut.xyz2sph(aa[0, nn ])
   idx = np.argwhere(self.point_cloud.dist_intp_coord_axis1[0, nn] != 1e10)
   n = len(idx)

   bb = np.zeros([n+1, 3])
   bb[:n, 0], bb[:n, 1], bb[:n, 2] = phi_[:n], theta_[:n], nn_u_axis1[0, nn, :n]

   rr, phii, thetaa = self.ut.xyz2sph(self.point_cloud.intp_coord_axis1[0])
   bb[-1, 0], bb[-1, 1], bb[-1, 2] = phii[nn], thetaa[nn], intp_u_axis1[0, nn]

   cbar=ax.scatter(bb[:, 0], bb[:, 1], c=bb[:,2], vmin=0.67, vmax=0.7)
   for i, txt in enumerate(bb[:,2]):
      ax.annotate(round(txt, 4), (bb[i, 0], bb[i, 1]))
   ax.scatter(bb[-1, 0], bb[-1, 1], color='none', edgecolor='k')
   plt.title(nn)
   plt.waitforbuttonpress()
plt.colorbar(cbar)





