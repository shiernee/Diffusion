import os
import sys
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import unittest
import numpy as np
import pandas as pd
from src.utils.RbfInterpolator import RbfInterpolatorSph, RbfInterpolatorCartesian
from src.utils.ExactInterpolator4test import ExactInterpolator4test
from src.pointcloud.PointCloud import PointCloud
from src.variables.LinearOperator import LinearOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.BaseVariables import BaseVariables
from src.variables.Variables import Variables
from src.variables.FiniteDiff2ndOrder import FiniteDiff2ndOrder
from src.variables.FiniteDiff4thOrder import FiniteDiff4thOrder
from src.utils.DataFrame import DataFrame
from Utils.Utils import xyz2r_phi_theta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================================================
class Testing(unittest.TestCase):
    print('Test Diffusion Code')
    print('A unit Sphere of 1000 points, u0 = sin(theta)**2*cos(phi) \n')

    def test_eval(self):
        test_linearoperator()
        test_cubicoperator()
        test_get_grid_list()
        test_local_axis_perpendicular(1e-5)
        test_make_grid(1e-5)
        test_diffusionoperator(1e-2)
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


# ======================================================
def test_get_grid_list():
    u0, coord = get_variable()
    pt_cld = PointCloud(coord, 0.05)

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


# ======================================================
def test_make_grid(error_limit):
    u0, coord = get_variable()
    pt_cld = PointCloud(coord, 0.05)
    spacing = pt_cld.interpolated_spacing

    err = False
    for idx in range(pt_cld.no_pt):
        # check local_grid midpt must be same as coordinate
        local_grid = pt_cld.local_grid[idx]
        mididx = int(len(local_grid) / 2)
        midpt = local_grid[mididx, mididx, :]
        coord = pt_cld.coord[idx]
        np.testing.assert_array_equal(coord, midpt,
                                      'local_grid midpt does not match coodinate at row {}'.format(idx))

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

    pt_cld = PointCloud(coord, 0.05)
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
def test_diffusionoperator(error_limit):
    print('========= start test_diffusionoperator ==========')

    err = False

    u0, coord = get_variable()
    D = np.ones(u0.shape)

    pt_cld = PointCloud(coord, 0.05)
    print('interpolated_spacing = ', pt_cld.interpolated_spacing)

    finite_diff = FiniteDiff2ndOrder()
    # finite_diff = FiniteDiff4thOrder()  #l_infinity reduced by 0.3

    interp = RbfInterpolatorSph(pt_cld)

    u = Variables(pt_cld, interp, finite_diff, 0)
    u.set_val(u0)

    x, y, z = pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2]
    r, phi, theta = xyz2r_phi_theta(x, y, z)

    # calculate diffusion error
    diff_op = DiffusionOperator(D)
    divu = diff_op.eval(u)
    diff_op_exact = (2 * np.cos(2 * theta) - 1) * np.cos(phi)
    diffusion_err = abs(diff_op_exact - divu) / diff_op_exact

    if diff_op_exact.max() > error_limit:
        err = True

    # calculate interpolation error in grid
    no_nn_interp_diff_diff_err_list = []
    no_neighbour_mean_dist_list = []
    for idx, grid in enumerate(pt_cld.get_grid_list()):
        gridval = interp.eval(grid, u.get_val())
        local_grid = grid[0].reshape([-1, 3])
        x, y, z = local_grid[:, 0], local_grid[:, 1], local_grid[:, 2]
        r2, phi2, theta2 = xyz2r_phi_theta(x, y, z)
        u0_exact2 = np.sin(theta2) * np.sin(theta2) * np.cos(phi2)  # for every point in grid
        intp_diff_tmp = abs(u0_exact2 - gridval.reshape([-1, ]))
        # print('pt_cld.nn_indices', pt_cld.nn_indices[idx])

        no_neighbour_tmp = len(pt_cld.nn_indices[idx])
        # print('no_neighbour_tmp ', no_neighbour_tmp)

        no_neighbour_grid = np.ones(intp_diff_tmp.shape) * no_neighbour_tmp
        # print('no_neighbour_grid ', no_neighbour_grid)

        neighbour_dist_mean_tmp = np.mean(pt_cld.dist_nn[idx])
        # print('neighbour_dist_grid ', neighbour_dist_mean_tmp)

        if intp_diff_tmp.max() > error_limit:
            err = True

        no_nn_interp_diff_diff_err_list.append([no_neighbour_grid, intp_diff_tmp])
        no_neighbour_mean_dist_list.append([no_neighbour_tmp, neighbour_dist_mean_tmp])

    interp_diff_no_nn_list = np.asarray(no_nn_interp_diff_diff_err_list)
    no_neighbour_mean_dist_list = np.asarray(no_neighbour_mean_dist_list)

    no_nn_grid = interp_diff_no_nn_list[:, 0]
    intp_diff_grid = interp_diff_no_nn_list[:, 1]

    no_neighbour = no_neighbour_mean_dist_list[:, 0]
    neighbour_dist_mean = no_neighbour_mean_dist_list[:, 1]

    intp_diff_grid_max = np.max(intp_diff_grid, axis=1)
    intp_diff_grid_mean = np.mean(intp_diff_grid, axis=1)

    idx_highest_diff_err = abs(diffusion_err).argmax()
    print('at points with highest diffusion error', diffusion_err[idx_highest_diff_err])
    print('local_grid', pt_cld.get_grid_list()[idx_highest_diff_err][0].reshape([-1, 3]))
    print('no_neighbour ', no_neighbour[idx_highest_diff_err])
    print('neighbour_dist ', pt_cld.dist_nn[idx_highest_diff_err])
    print('neighbour_dist_mean ', neighbour_dist_mean[idx_highest_diff_err])
    print('intp_diff_grid ', intp_diff_grid[idx_highest_diff_err])
    print('intp_diff_grid_mean ', intp_diff_grid_mean[idx_highest_diff_err])

    if err is True:
        print('mean diff_err: {}'.format(np.mean(diffusion_err)))
        print('max diff_err: {}'.format(np.max(diffusion_err)))
        print('mean interp_diff ', np.mean(intp_diff_grid))
        print('max interp_diff ', np.max(intp_diff_grid))
        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.plot(no_neighbour, diffusion_err, '.')
        ax1.set_xlabel('no_neighbour')
        ax1.set_ylabel('diffusion_err')
        ax2 = fig.add_subplot(322)
        ax2.plot(no_nn_grid, intp_diff_grid, '.')
        ax2.set_xlabel('no_neighbour')
        ax2.set_ylabel('intp_diff_grid')
        ax3 = fig.add_subplot(323)
        ax3.plot(intp_diff_grid_mean, diffusion_err, '.')
        ax3.set_xlabel('intp_diff_grid_mean')
        ax3.set_ylabel('diffusion_err')
        ax4 = fig.add_subplot(324)
        ax4.plot(diffusion_err, neighbour_dist_mean, '.')
        ax4.set_xlabel('diffusion_err')
        ax4.set_ylabel('neighbour_dist_mean')
        ax5 = fig.add_subplot(325)
        ax5.plot(no_neighbour, neighbour_dist_mean, '.')
        ax5.set_xlabel('no_neighbour')
        ax5.set_ylabel('neighbour_dist_mean')


        fig2 = plt.figure()
        ax22 = fig2.add_subplot(111, projection='3d')
        cbar = ax22.scatter(pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2],
                     c=diffusion_err, cmap='jet')
        ax22.set_title('diffusion_err')
        fig2.colorbar(cbar)

        fig3 = plt.figure()
        ax33 = fig3.add_subplot(111, projection='3d')
        cbar = ax33.scatter(pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2],
                            c=intp_diff_grid_mean, cmap='jet')
        ax33.set_title('intp_diff_grid_mean')
        fig3.colorbar(cbar)

        plt.show()
        raise ValueError('error of divu larger than {}'.format(error_limit))

    print('test_diffusionoperator done == passed')
    sys.stdout.flush()
    return

'''
# ======================================================
def test_interp(error_limit):
    print('========= start test_interp ==========')

    err = False

    u0, coord = get_variable()

    pt_cld = PointCloud(coord, 0.05)
    finite_diff = FiniteDiff2ndOrder()
    interp = RbfInterpolatorSph(pt_cld)
    # interp = ExactInterpolator4test(pt_cld)
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

    interp = RbfInterpolatorSph(pt_cld)
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
    dataframe = DataFrame("database.csv")
    coord = dataframe.get_coord()
    u0 = dataframe.get_uni_u()
    return u0, coord




# ====================================================
if __name__ == '__main__':
    unittest.main()

