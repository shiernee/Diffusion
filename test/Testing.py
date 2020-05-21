import os
import sys
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import unittest
import math
import numpy as np
import pandas as pd
from src.utils.RbfInterpolator import RbfInterpolatorSphCubic, RbfInterpolatorCartesian, RbfInterpolatorIDW, \
    RbfInterpolatorGaussian, NearestNBInterpolator, LinearNBInterpolator, IDWNBInterpolator
from src.variables.QuadraticFormDivCalculator import QuadraticFormDivCalculator
from src.variables.DiffusionOperatorFitMethod import DiffusionOperatorFitMethod
from src.utils.IDWInterpolator import IDWInterpolator
from src.utils.ExactInterpolator4test import ExactInterpolator4test
from src.pointcloud.PointCloud import PointCloud
from src.variables.LinearOperator import LinearOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.BaseVariables import BaseVariables
from src.variables.Variables import Variables
from src.utils.DataFrame import DataFrame
from Utils.Utils import xyz2r_phi_theta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ====================================================
class Testing(unittest.TestCase):

    def test_eval(self):
        test_linearoperator()
        test_cubicoperator()
        test_IDWInterpolator()
        test_get_grid_list()
        test_QuadraticFormDivCalculator()
        test_local_axis_perpendicular(1e-5)
        test_make_grid(1e-5)
        test_diffusionoperator_error(1e-5)
        # test_interp(1e-2)  # trouble
        # test_u0(1e-5)
        # test_plot_intp_u0(1e-5)

        return


# =======================================================
def test_linearoperator():
    u = BaseVariables(t=0)
    u.set_val(10)

    lin_op = LinearOperator(4)
    assert lin_op.eval(u) == 40, 'lin_op.eval != 40'
    print('test_linearoperator done == passed')
    sys.stdout.flush()


# =======================================================
def test_cubicoperator():
    u = BaseVariables(t=0)
    u.set_val(10)

    cub_op = CubicOperator(3)
    assert cub_op.eval(u) == -630, 'cub_op.eval(u) != -630'
    print('test_cubicoperator done == passed')
    sys.stdout.flush()


def test_IDWInterpolator():
    x = np.array([[0.5032176, 0.56230778, 0.11241475],
                  [0.37173495, 0.7693895, 0.15154518],
                  [0.0227865, 0.60131664, 0.49170493],
                  [0.63170145, 0.05394829, 0.37977266],
                  [0.37985933, 0.70971671, 0.46411566],
                  [0.2683475, 0.03140954, 0.13891893],
                  [0.11318219, 0.76351362, 0.07024305],
                  [0.7910568, 0.15224475, 0.47943285],
                  [0.83260074, 0.56536564, 0.97109803],
                  [0.47185391, 0.96013915, 0.23200289]])
    y = np.array([9.14866715, 2.55358226, 4.53695721,
                  3.63892562, 5.62148883, 4.90972287,
                  3.09452405, 6.35390295, 2.36381706,
                  5.88413416])

    x_target = np.array([[0.756279445, 0.146581044, 0.394003803],
                         [0.75879349, 0.996188697, 0.192774222],
                         [0.181619578, 0.448327672, 0.032964805],
                         [0.47185391, 0.96013915, 0.23200289]])

    interp = IDWInterpolator(x, y)
    interp_value = interp(x_target)
    exact_intp_value = np.array([5.60824383, 5.275697172, 4.980061723, 5.88413416])
    np.testing.assert_almost_equal(interp_value, exact_intp_value)
    print('test_IDWInterpolator done == passed')

    return


# ======================================================
def test_get_grid_list():
    u0, coord = get_variable()
    pt_cld = PointCloud(coord, 0.05, 3)

    for idx, grid in enumerate(pt_cld.get_grid_list()):
        grid_list = grid[0]
        local_grid = pt_cld.local_grid[idx]
        np.testing.assert_array_equal(grid_list, local_grid)

        nn_indices = grid[1]
        dist_nn = grid[2]
        assert nn_indices.shape == dist_nn.shape, \
            'grid[1] and grid[2] must have same shape'
    print('test_get_grid_list done == passed')
    return


def test_QuadraticFormDivCalculator():
    quadform = QuadraticFormDivCalculator()

    data = np.array([[0.21347281, -0.08514052, 0.37463643],
                     [-0.15028533, -0.51710984, -0.35407931],
                     [2.5918318, 0.19476291, 3.77055497],
                     [-0.51093292, -1.61567322, -1.43341006],
                     [2.31353337, -1.26856376, 1.37872144],
                     [0.41001983, -0.72827269, -0.20773432],
                     [0.99719954, -1.93068371, -0.00596601],
                     [-1.54014393, 1.6333638, -0.3520338],
                     [-0.40606095, -0.10822079, -0.83447536],
                     [-0.66210132, -1.18771663, -1.3089941]])

    linear_coeff = np.asarray([0.15599342, 1.15863002, 0.76708283])
    quad_coeff = np.array([0.06651248, 1.37818442, 1.0410777, 0.28704309, -0.03431101, \
                           0.29896695])

    grid2D = data[:, :2]
    val = data[:, 2]
    d = quadform.fit_linear(grid2D, val)
    c = quadform.fit_quadratic(grid2D, val)

    np.testing.assert_almost_equal(d, linear_coeff)
    np.testing.assert_almost_equal(c, quad_coeff)

    print('test_QuadraticFormDivCalculator done == passed')
    return


# ======================================================
def test_make_grid(error_limit):
    u0, coord = get_variable()
    grid_length = 0.1
    local_grid_resolution = 3
    pt_cld = PointCloud(coord, grid_length, local_grid_resolution)

    spacing = grid_length / (pt_cld.local_grid_resolution - 1)

    err = False
    for idx in range(pt_cld.no_pt):
        # check local_grid midpt must be same as coordinate
        local_grid = pt_cld.local_grid[idx]
        mididx = int(len(local_grid) / 2)
        midpt = local_grid[mididx, mididx, :]
        coord = pt_cld.coord[idx]
        np.testing.assert_array_equal(coord, midpt,
                                      'local_grid midpt does not match coodinate at row {}'.format(idx))

        # check local_grid_length must be equal to grid_length
        top = round(np.linalg.norm(local_grid[0, -1] - local_grid[0, 0]), 8)
        left = round(np.linalg.norm(local_grid[-1, 0] - local_grid[0, 0]), 8)
        right = round(np.linalg.norm(local_grid[-1, -1] - local_grid[0, -1]), 8)
        bottom = round(np.linalg.norm(local_grid[-1, -1] - local_grid[-1, 0]), 8)
        if (top != grid_length) or (left != grid_length) or (right != grid_length) or (bottom != grid_length):
            print('top: {}, left: {}, right: {}, bottom: {}'.format(top, left, right, bottom))
            print('grid_length generated does not match with grid_length specified')
            err = True

        # check grid spacing must be equal to pt_cld.interpolated_spacing
        leftpt = local_grid[mididx, mididx - 1, :]
        rightpt = local_grid[mididx, mididx + 1, :]
        toppt = local_grid[mididx - 1, mididx, :]
        bottompt = local_grid[mididx + 1, mididx, :]
        sp = []
        sp.append(np.linalg.norm(midpt - leftpt))
        sp.append(np.linalg.norm(midpt - rightpt))
        sp.append(np.linalg.norm(midpt - toppt))
        sp.append(np.linalg.norm(midpt - bottompt))
        sp = np.asarray(sp)
        sp_value = abs(spacing * np.ones(sp.shape) - sp).max()
        if sp_value > error_limit * spacing:
            print('sp is ', sp)
            print('spacing error {:.6f} at {}, true spacing {:.6f}'.format(sp_value, idx, spacing))
            err = True

        # check that all points on grid are on same plane
        normal_vector = np.cross((midpt - leftpt), (midpt - toppt))
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        dotary = np.dot(local_grid.reshape([-1, 3]), normal_vector)
        planar_value = abs(dotary[0] * np.ones(dotary.shape) - dotary).max()
        if planar_value > error_limit:
            print('grid not on plane {} at {}'.format(planar_value, idx))
            err = True

    if err is True:
        raise ValueError('test_make_grid error')
    print('test_make_grid done == passed')
    return


# =======================================================
def test_local_axis_perpendicular(error_limit):
    u0, coord = get_variable()

    pt_cld = PointCloud(coord, 0.05, 3)
    dotprod = []
    for i in range(pt_cld.no_pt):
        local_axis1_tmp = pt_cld.local_axis1[i].squeeze()
        local_axis2_tmp = pt_cld.local_axis2[i].squeeze()
        dotprod_tmp = np.dot(local_axis1_tmp, local_axis2_tmp)
        dotprod.append(dotprod_tmp)

    if all(i <= error_limit for i in dotprod) is False:
        raise ValueError('local_axis1 and local_axis2 must be perpendicular')
    print('test_local_axis_perpendicular done == passed')
    sys.stdout.flush()
    return


# =======================================================
def test_diffusionoperator_error(error_limit):
    print('========= start test_diffusionoperator ==========')

    err = True

    u0, coord = get_variable()
    grid_length = 0.2
    local_grid_resolution = 21
    pt_cld = PointCloud(coord, grid_length, local_grid_resolution)
    print('interpolated_spacing = ', pt_cld.interpolated_spacing)

    interp = RbfInterpolatorSphCubic(pt_cld)
    # interp = RbfInterpolatorIDW(pt_cld)
    # interp = RbfInterpolatorGaussian(pt_cld)
    # interp = NearestNBInterpolator(pt_cld)
    # interp = LinearNBInterpolator(pt_cld)
    # interp = IDWNBInterpolator(pt_cld)

    u = Variables(pt_cld, interp, 0)
    u.set_val(u0)

    D = Variables(pt_cld, interp, 0)
    D.set_val(np.ones(u0.shape))

    x, y, z = pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2]
    r, phi, theta = xyz2r_phi_theta(x, y, z)

    # calculate diffusion error
    # diff_op = DiffusionOperator(D)
    diff_op = DiffusionOperatorFitMethod(D, pt_cld)
    divu = diff_op.eval(u, interp)
    diff_op_exact = (2 * np.cos(2 * theta) - 1) * np.cos(phi)
    diffusion_err = abs(diff_op_exact - divu)
    log_diffusion_err = np.log10(diffusion_err)

    if diff_op_exact.max() > error_limit:
        err = True

    # calculate interpolation error in grid
    no_nn_interp_diff_list = []
    no_neighbour_mean_dist_list = []
    for idx in range(pt_cld.no_pt):
        grid = pt_cld.get_grid_list()[idx]
        gridval = interp.eval(grid, u.get_val())
        local_grid = grid[0].reshape([-1, 3])
        x, y, z = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        r2, phi2, theta2 = xyz2r_phi_theta(x, y, z)
        u0_exact2 = np.sin(theta2) * np.sin(theta2) * np.cos(phi2)  # for every point in grid

        intp_diff_tmp = abs(u0_exact2 - gridval.reshape([-1, ]))
        no_neighbour_tmp = len(pt_cld.nn_indices[idx])
        no_neighbour_grid = np.ones(intp_diff_tmp.shape) * no_neighbour_tmp
        neighbour_dist_mean_tmp = np.mean(pt_cld.dist_nn[idx])

        if intp_diff_tmp.max() > error_limit:
            err = True

        no_nn_interp_diff_list.append([no_neighbour_grid, intp_diff_tmp])
        no_neighbour_mean_dist_list.append([no_neighbour_tmp, neighbour_dist_mean_tmp])

    no_nn_interp_diff_list = np.asarray(no_nn_interp_diff_list)
    no_neighbour_mean_dist_list = np.asarray(no_neighbour_mean_dist_list)

    no_nn_grid = no_nn_interp_diff_list[:, 0]
    intp_diff_grid = no_nn_interp_diff_list[:, 1]
    log_intp_diff_grid = [np.log10(i) for i in no_nn_interp_diff_list[:, 1]]

    no_neighbour = no_neighbour_mean_dist_list[:, 0]
    neighbour_dist_mean = no_neighbour_mean_dist_list[:, 1]

    intp_diff_grid_max = np.max(intp_diff_grid, axis=1)
    intp_diff_grid_mean = np.mean(intp_diff_grid, axis=1)

    if err is True:
        print('mean diff_err: {}'.format(np.nanmean(diffusion_err)))
        print('max diff_err: {}'.format(np.nanmax(diffusion_err)))
        print('mean interp_diff ', np.nanmean(intp_diff_grid))
        print('max interp_diff ', np.nanmax(intp_diff_grid))

        # =====================================================
        # plot log diffusion error vs no of neighbour
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(no_neighbour, log_diffusion_err, '.')
        ax0.set_xlabel('no_neighbour')
        ax0.set_ylabel('log_diffusion_err')
        ax0.set_ylim(-4, 1)
        ax0.grid()

        # =====================================================
        # plot log diffusion error vs theta vs phi
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cbar = ax1.scatter(theta, phi, c=log_diffusion_err, cmap='jet',
                           s=7, vmin=-4, vmax=1)
        ax1.set_xlabel('theta')
        ax1.set_ylabel('phi')
        ax1.set_xlim(-0.1, 3.2)
        ax1.set_title('log_diffusion_err')
        ax1.grid()
        fig.colorbar(cbar, ax=ax1)

        # =====================================================
        nn_range = [40, 70]
        condition1 = no_neighbour < nn_range[0], 'no_neighbour < {}'.format(nn_range[0])
        condition2 = ((nn_range[0] < no_neighbour) & (no_neighbour < nn_range[1])), \
                     '(({} < no_neighbour) & (no_neighbour < {}))'.format(nn_range[0], nn_range[1])
        condition3 = no_neighbour > nn_range[1], 'no_neighbour > {}'.format(nn_range[1])

        # plot log diffusion error vs theta vs phi for different no of neighbour
        fig = plt.figure()
        ax2 = fig.add_subplot(231)
        cbar = ax2.scatter(theta[condition1[0]], phi[condition1[0]], c=log_diffusion_err[condition1[0]],
                           cmap='jet', s=7, vmin=-4, vmax=1)
        ax2.set_xlabel('theta')
        ax2.set_ylabel('phi')
        ax2.set_xlim(-0.1, 3.2)
        ax2.set_title(condition1[1])
        ax2.grid()
        fig.colorbar(cbar, ax=ax2)

        ax3 = fig.add_subplot(232)
        cbar = ax3.scatter(theta[condition2[0]], phi[condition2[0]], c=log_diffusion_err[condition2[0]],
                           cmap='jet', s=7, vmin=-4, vmax=1)
        ax3.set_xlabel('theta')
        ax3.set_ylabel('phi')
        ax3.set_xlim(-0.1, 3.2)
        ax3.set_title(condition2[1])
        ax3.grid()
        fig.colorbar(cbar, ax=ax3)

        ax4 = fig.add_subplot(233)
        cbar = ax4.scatter(theta[condition3[0]], phi[condition3[0]], c=log_diffusion_err[condition3[0]],
                           cmap='jet', s=7, vmin=-4, vmax=1)
        ax4.set_xlabel('theta')
        ax4.set_ylabel('phi')
        ax4.set_xlim(-0.1, 3.2)
        ax4.set_title(condition3[1])
        ax4.grid()
        fig.colorbar(cbar, ax=ax4)

        fig.tight_layout()

        # =======================================================================
        plt.figure()
        n=0
        for i in range(nn_range[0], nn_range[-1], 5):
            n += 1
            plt.subplot('32{}'.format(n))
            plt.hist(log_diffusion_err[no_neighbour == i], bins=30,  edgecolor='k')
            plt.xlim(-4, 1)
            plt.xlabel('log_diffusion_err')
            plt.ylabel('count')
            plt.title('no_neighbor {}'.format(i))
        plt.tight_layout()
        '''
        ####################################################################
        ## ===== plot log_diffusion_err vs no_neighbour at region =====   ##
        ## ===== region with [abs(u0) > 0.05] =======                       ##
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(no_neighbour[abs(u0) > 0.05], log_diffusion_err[abs(u0) > 0.05], '.')
        ax0.set_xlabel('no_neighbour')
        ax0.set_ylabel('log_diffusion_err')
        ax0.set_ylim(-4, 1)
        ax0.set_title('log_diffusion_err exclude u0 near zero')
        ax0.grid()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cbar = ax1.scatter(theta[abs(u0) > 0.05], phi[abs(u0) > 0.05],
                           c=log_diffusion_err[abs(u0) > 0.05],
                           cmap='jet', s=7, vmin=-4, vmax=1)
        ax1.set_xlabel('theta')
        ax1.set_ylabel('phi')
        ax1.set_xlim(-0.1, 3.2)
        ax1.set_title('log_diffusion_err exclude u0 near zero')
        ax1.grid()
        fig.colorbar(cbar, ax=ax1)

        ####################################################################
        ## ===== plot log_diffusion_err vs no_neighbour at region =====   ##
        ## ===== region with no_neighbour < 100 & log_diffusion_err < -1 ====        ##
        condition = ((no_neighbour < 100) & (log_diffusion_err < -1))
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(no_neighbour[condition],
                 log_diffusion_err[condition],
                 '.')
        ax0.set_xlabel('no_neighbour')
        ax0.set_ylabel('log_diffusion_err')
        ax0.set_ylim(-4, 1)
        ax0.set_title('(no_neighbour < 100) & (log_diffusion_err < -1)')
        ax0.grid()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cbar = ax1.scatter(theta[condition], phi[condition], c=log_diffusion_err[condition],
                           cmap='jet', s=7, vmin=-4, vmax=1)
        ax1.set_xlabel('theta')
        ax1.set_ylabel('phi')
        ax1.set_xlim(-0.1, 3.2)
        ax1.set_title('(no_neighbour < 100) & (log_diffusion_err < -1)')
        ax1.grid()
        fig.colorbar(cbar, ax=ax1)

        #######################################################################
        ####### ----===== plot u0 vs theta vs phi =====      ##################
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cbar = ax1.scatter(theta, phi, c=u0, cmap='jet',
                           s=7)
        ax1.set_xlabel('theta')
        ax1.set_ylabel('phi')
        ax1.set_xlim(-0.1, 3.2)
        ax1.set_title('u0')
        ax1.grid()
        fig.colorbar(cbar, ax=ax1)

        # =====================================================
        # plot log diffusion error in 3D
        fig2 = plt.figure()
        ax22 = fig2.add_subplot(111, projection='3d')
        cbar = ax22.scatter(pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2],
                            c=log_diffusion_err, cmap='jet')
        ax22.set_title('diffusion_err')
        fig2.colorbar(cbar)

        # =====================================================
        # plot u0 in 3D
        fig2 = plt.figure()
        ax22 = fig2.add_subplot(111, projection='3d')
        cbar = ax22.scatter(pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2],
                            c=u0, cmap='jet')
        ax22.set_title('u0')
        fig2.colorbar(cbar)

        # =====================================================
        # plot intp_diff_grid_mean in 3D
        fig3 = plt.figure()
        ax33 = fig3.add_subplot(111, projection='3d')
        cbar = ax33.scatter(pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2],
                            c=intp_diff_grid_mean, cmap='jet')
        fig3.colorbar(cbar)
        ax33.set_title('intp_diff_grid_mean')
        '''
        plt.show()

        # == raise ValueError =================
        raise ValueError('error of divu larger than {}'.format(error_limit))

    print('test_diffusionoperator done == passed')
    sys.stdout.flush()
    return


def test_u_value_at_time_n(error_limit):
'''
# ======================================================
def test_interp(error_limit):
    print('========= start test_interp ==========')

    err = False

    u0, coord = get_variable()

    pt_cld = PointCloud(coord, 0.05)
    finite_diff = FiniteDiff2ndOrder()
    
    # interp = ExactInterpolator4test(pt_cld)    
    interp = RbfInterpolatorSphCubic(pt_cld)
    # interp = RbfInterpolatorIDW(pt_cld)
    # interp = RbfInterpolatorGaussian(pt_cld)
    # interp = NearestNBInterpolator(pt_cld)
    # interp = LinearNBInterpolator(pt_cld)
    # interp = IDWNBInterpolator(pt_cld)
    
    u = Variables(pt_cld, interp, finite_diff, 0)
    u.set_val(u0)
    print('interpolate spacing ', pt_cld.interpolated_spacing)

    interp_diff_no_nn_list = []

    for idx, grid in enumerate(pt_cld.get_grid_list()):
        gridval = interp.eval(grid, u.get_val())
        local_grid = grid[0].reshape([-1, 3])
        x, y, z = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        r2, phi2, theta2 = xyz2r_phi_theta(x, y, z)
        u0_exact2 = np.sin(theta2) * np.sin(theta2) * np.cos(phi2)  # for every point in grid
        intp_diff_tmp = abs(u0_exact2 - gridval.reshape([-1, ]))
        nn_indices = (np.ones(intp_diff_tmp.shape) * len(pt_cld.nn_indices[idx]))
        if intp_diff_tmp.max() > error_limit:
            err = True

        interp_diff_no_nn_list.append([intp_diff_tmp, nn_indices])

    interp_diff_no_nn_list = np.asarray(interp_diff_no_nn_list)
    intp_diff = interp_diff_no_nn_list[:, 0]
    no_nn = interp_diff_no_nn_list[:, 1]

    if err is True:
        print('mean interp_diff ', np.mean(intp_diff))
        print('max interp_diff ', np.max(intp_diff))
        plt.figure()
        plt.plot(no_nn, intp_diff, '.')
        plt.xlabel('number of neighbour points')
        plt.ylabel('intp_diff')
        plt.figure()
        plt.hist(intp_diff, bins=50)
        plt.xlabel('count')
        plt.ylabel('intp_diff')
        plt.show()
        raise ValueError('error in interpolation more than ', error_limit)

    print('test_interp done == passed')
    sys.stdout.flush()
    return


# =============================================================
def test_u0(error_limit):
    print('========= start test_u0 ==========')
    u0, coord, param_file, dataframe = get_variable()

    pt_cld = PointCloud(coord, 0.05)
    finite_diff = FiniteDiff2ndOrder()

    interp = RbfInterpolatorSphCubic(pt_cld)
    # interp = RbfInterpolatorIDW(pt_cld)
    u = Variables(pt_cld, interp, finite_diff, 0)
    u.set_val(u0)
    x, y, z = pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2]
    r, phi, theta = xyz2r_phi_theta(x, y, z)
    u0_exact = np.sin(theta) * np.sin(theta) * np.cos(phi)

    l_infinity = abs(u0_exact - u.get_val()).max()
    if l_infinity > error_limit:
        print('intrp error {}'.format(l_infinity))
        raise ValueError('l_infinity_norm of u.get_val() larger than {}'.format(error_limit))

    print('test_u0 done == passed')
    sys.stdout.flush()

# ====================================================
def test_plot_intp_u0():
    u0, coord, param_file, dataframe = get_variable()
    pt_cld = PointCloud(coord, 0.05)
    finite_diff = FiniteDiff2ndOrder()

    interp = RbfInterpolatorCartesian(pt_cld)
    # interp = RbfInterpolatorIDW(pt_cld)
    u = Variables(pt_cld, interp, finite_diff, 0)
    u.set_val(u0)

    gridval_list=[]
    intp_cood_list=[]
    for idx, grid in enumerate(pt_cld.get_grid_list()):
        gridval = interp.eval(grid, u.get_val())
        gridval_list.append(gridval.reshape([-1, ]))
        intp_cood_list.append(grid[0].reshape([-1, 3]))
    gridval_list = np.asarray(gridval_list).reshape([-1, ])
    intp_cood_list = np.asarray(intp_cood_list).reshape([-1, 3])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cbar = ax.scatter(intp_cood_list[:, 0], intp_cood_list[:, 1], intp_cood_list[:, 2], c=gridval_list)
    plt.colorbar(cbar)
    plt.show()
    return
'''

# =======================================================
def get_variable():
    dataframe = DataFrame("database5000_normdist.csv")
    coord = dataframe.get_coord()
    u0 = dataframe.get_uni_u()

    print('Test Diffusion Code')
    print('A unit Sphere of {} points, u0 = sin(theta)**2*cos(phi)'.format(len(coord)))

    return u0, coord


# ====================================================
if __name__ == '__main__':
    unittest.main()
