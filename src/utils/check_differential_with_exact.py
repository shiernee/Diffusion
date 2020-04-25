'''
this code is to check different method of computing dudt at t=0 and compared with solution obtained from formula.

steps:
1. debuug liang_et_al_main.py till Diffusion_Model.py line " shape_to_be_taken = ...." run this line also.
2. copy past below 
created on 19 Apr 2020 by snsaw
'''


# ===== compute dudt at t=0
# Section 1: get exact_u value using u(t, theta) = exp(-2*t)*cos(theta) with three different methods
# compute dudt using exact_u value - this is to check which methods give lowest error; this part eliminitate the error induced by
# interpolaton 
t=0
exact_u = []
for i in range(self.point_cloud.no_pt):
     intp_coord = np.concatenate((self.point_cloud.intp_coord_axis1[i],
                                  self.point_cloud.intp_coord_axis2[i]),
                                  axis=0)
     rintp, phiintp, thetaintp = self.ut.xyz2sph(intp_coord)
     exact_u.append(np.exp(-2*t)*np.cos(thetaintp))

# -- method 1---
exact_u = np.array(exact_u)
exact_u_axis1 = exact_u[:, :int(len(intp_coord)/2)]
exact_u_axis2 = exact_u[:, int(len(intp_coord)/2):]

exact_dudx = np.matmul(exact_u_axis1, coeff_matrix_first_der) / interpolated_spacing
exact_dudy = np.matmul(exact_u_axis2, coeff_matrix_first_der) / interpolated_spacing
exact_divD_dudx = np.multiply(exact_dudx, intp_D_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]])
exact_divD_dudy = np.multiply(exact_dudy, intp_D_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]])
exact_term1 = np.matmul(exact_divD_dudx, coeff_matrix_second_der) / interpolated_spacing
exact_term2 = np.matmul(exact_divD_dudy, coeff_matrix_second_der) / interpolated_spacing
exact_dDdV_dx2 = np.squeeze(exact_term1 + exact_term2) + c
exact_laplacian = exact_dDdV_dx2.copy()

# -- method 2 - D is scalar---
coeff_matrix = np.zeros([exact_u_axis1.shape[1], 1])
mid = int(exact_u_axis1.shape[1] / 2)
coeff = np.array([1, -2, 1]).reshape([-1,1])
coeff_matrix[mid-int(len(coeff)/2):mid+int(len(coeff)/2)+1]=coeff
exact_dudx2 = np.matmul(exact_u_axis1, coeff_matrix)/interpolated_spacing
exact_dudy2 = np.matmul(exact_u_axis2, coeff_matrix)/interpolated_spacing
exact_dDdV_dx2_2 = np.squeeze(exact_dudx2 + exact_dudy2) + c
exact_laplacian2 = exact_dDdV_dx2_2.copy()

# -- method 3 - D is a function of r(x,y,z) ---
coeff_matrix = np.zeros([exact_u_axis1.shape[1], 1])
mid = int(exact_u_axis1.shape[1] / 2)
#coeff = np.array([-1/12, 4/3, -5/2, 4/3, -1/12]).reshape([-1,1])
coeff = np.array([1, -2, 1]).reshape([-1,1])
coeff_matrix[mid-int(len(coeff)/2):mid+int(len(coeff)/2)+1]=coeff

D_prime_dx = np.matmul(intp_D_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
D_prime_dy = np.matmul(intp_D_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
exact_u_prime_dx = np.matmul(exact_u_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
exact_u_prime_dy = np.matmul(exact_u_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
exact_u_pp_dx = np.matmul(exact_u_axis1, coeff_matrix)/interpolated_spacing
exact_u_pp_dy = np.matmul(exact_u_axis2, coeff_matrix)/interpolated_spacing

D = self.D.reshape([-1,1])
exact_term1_3 = D_prime_dx * exact_u_prime_dx + D_prime_dy * exact_u_prime_dy
exact_term2_3 = D * exact_u_pp_dx + D * exact_u_pp_dy
exact_laplacian3 = np.squeeze(exact_term1_3 + exact_term2_3) + c

# Section 2: 
# compute dudt using intp_u value using 3 different methods

# -- numerical method 1---
dudx = np.matmul(intp_u_axis1, coeff_matrix_first_der) / interpolated_spacing
dudy = np.matmul(intp_u_axis2, coeff_matrix_first_der) / interpolated_spacing
divD_dudx = np.multiply(dudx, intp_D_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]])
divD_dudy = np.multiply(dudy, intp_D_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]])
term1 = np.matmul(divD_dudx, coeff_matrix_second_der) / interpolated_spacing
term2 = np.matmul(divD_dudy, coeff_matrix_second_der) / interpolated_spacing
dDdV_dx2 = np.squeeze(term1 + term2) + c
num_laplacian =dDdV_dx2.copy()

# -- numerical method 2---
coeff_matrix = np.zeros([intp_u_axis1.shape[1], 1])
mid = int(intp_u_axis1.shape[1] / 2)
coeff = np.array([1, -2, 1]).reshape([-1,1])
coeff_matrix[mid-int(len(coeff)/2):mid+int(len(coeff)/2)+1]=coeff

num_dudx2 = np.matmul(intp_u_axis1, coeff_matrix)/interpolated_spacing
num_dudy2 = np.matmul(intp_u_axis2, coeff_matrix)/interpolated_spacing
num_dDdV_dx2_2 = np.squeeze(num_dudx2 + num_dudy2) + c
num_laplacian2 = num_dDdV_dx2_2.copy()

# -- numerical method 3---
coeff_matrix = np.zeros([intp_u_axis1.shape[1], 1])
mid = int(intp_u_axis1.shape[1] / 2)
coeff = np.array([-1/12, 4/3, -5/2, 4/3, -1/12]).reshape([-1,1])
coeff_matrix[mid-int(len(coeff)/2):mid+int(len(coeff)/2)+1]=coeff

D_prime_dx = np.matmul(intp_D_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
D_prime_dy = np.matmul(intp_D_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
num_u_prime_dx = np.matmul(intp_u_axis1[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
num_u_prime_dy = np.matmul(intp_u_axis2[:, shape_to_be_taken[0]:shape_to_be_taken[-1]], coeff_matrix_second_der) / interpolated_spacing
num_u_pp_dx = np.matmul(intp_u_axis1, coeff_matrix)/interpolated_spacing
num_u_pp_dy = np.matmul(intp_u_axis2, coeff_matrix)/interpolated_spacing

D = self.D.reshape([-1, 1])
num_term1_3 = D_prime_dx * num_u_prime_dx + D_prime_dy * num_u_prime_dy
num_term2_3 = D * num_u_pp_dx + D * num_u_pp_dy
num_laplacian3 = np.squeeze(num_term1_3 + num_term2_3)+ c

# Section 3: 
# compute dudt using formula, dudt = -2*exp(-2*t)*sin(theta)
# -- analytical solution --- 
r, phi, theta = self.ut.xyz2sph(self.point_cloud.coord)
formula_grad = -2*np.exp(-2*t)*np.sin(theta)
formula_laplacian = -2*np.exp(-2*t)*np.cos(theta)

# Section 4: compute error between exact_dudt and numerical_dudt with analytical solution 
err_exact = abs((exact_laplacian - formula_laplacian)/formula_laplacian)
err_exact2 = abs((exact_laplacian2 - formula_laplacian)/formula_laplacian)
err_exact3 = abs((exact_laplacian3 - formula_laplacian)/formula_laplacian)
err_num = abs((num_laplacian - formula_laplacian)/formula_laplacian)
err_num2 = abs((num_laplacian2 - formula_laplacian)/formula_laplacian)
err_num3 = abs((num_laplacian3 - formula_laplacian)/formula_laplacian)

print('max err_exact: {}'.format(err_exact.max()))
print('mean err_exact: {}'.format(err_exact.mean()))
print('max err_exact2: {}'.format(err_exact2.max()))
print('mean err_exact2: {}'.format(err_exact2.mean()))
print('max err_exact3: {}'.format(err_exact3.max()))
print('mean err_exact3: {}'.format(err_exact3.mean()))
print('-----------------------------')
print('max err_num: {}'.format(err_num.max()))
print('mean err_num: {}'.format(err_num.mean()))
print('max err_num2: {}'.format(err_num2.max()))
print('mean err_num2: {}'.format(err_num2.mean()))
print('max err_num3: {}'.format(err_num3.max()))
print('mean err_num3: {}'.format(err_num3.mean()))
      
         
