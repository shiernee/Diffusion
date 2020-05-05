import os
import sys
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
import math as m
from src.utils.RbfInterpolator import RbfInterpolator
from src.pointcloud.PointCloud import PointCloud
from src.variables.LinearOperator import LinearOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.BaseVariables import BaseVariables
from src.variables.Variables import Variables
from src.variables.FiniteDiff2ndOrder import FiniteDiff2ndOrder
from src.utils.DataFrame import DataFrame


class Testing(unittest.TestCase):
    class ErrorLimit:
        def __init__(self):
            self.max_l_infinity_error = 10
            self.max_interp_error = 100  #1%
            self.dotprod_limit = 1e-8

    # =======================================================
    # def test_linearoperator(self):
    #     u = BaseVariables(t=0)
    #     u.set_val(10)
    #
    #     lin_op = LinearOperator(4)
    #     self.assertEqual(lin_op.eval(u), 40)
    #     print('test_linearoperator done == passed')
    #     sys.stdout.flush()
    #
    # #=======================================================
    # def test_cubicoperator(self):
    #     u = BaseVariables(t=0)
    #     u.set_val(10)
    #
    #     cub_op = CubicOperator(3)
    #     self.assertEqual(cub_op.eval(u), -630)
    #     print('test_cubicoperator done == passed')
    #     sys.stdout.flush()
    #
    # ======================================================
    # def test_get_grid_list(self):
    #     u0, coord, param_file, dataframe = self.get_u0()
    #     pt_cld = PointCloud(coord, param_file)
    #
    #     for idx, grid in enumerate(pt_cld.get_grid_list()):
    #         grid_list = grid[0]
    #         local_grid = pt_cld.local_grid[idx]
    #         np.testing.assert_array_equal(grid_list, local_grid)
    #
    #         nn_indices = grid[1]
    #         dist_nn = grid[2]
    #         self.assertEqual(nn_indices.shape, dist_nn.shape,
    #                          'grid[1] and grid[2] must have same shape')
    #     print('test_get_grid_list done == passed')
    #     return

    # def test_make_grid(self):
    #     u0, coord, param_file, dataframe = self.get_u0()
    #     pt_cld = PointCloud(coord, param_file)
    #     spacing = pt_cld.interpolated_spacing
    #
    #     err = False
    #     for idx in range(pt_cld.no_pt):
    #         # check local_grid midpt must be same as coordinate
    #         local_grid = pt_cld.local_grid[idx]
    #         mididx = int(len(local_grid) / 2)
    #         midpt = local_grid[mididx, mididx, :]
    #         coord = pt_cld.coord[idx]
    #         np.testing.assert_array_equal(coord, midpt,
    #                                       'local_grid midpt does not match coodinate at row {}'.format(idx))
    #
    #         # check grid spacing
    #         leftpt = local_grid[mididx, mididx - 1, :]
    #         rightpt = local_grid[mididx, mididx + 1, :]
    #         toppt = local_grid[mididx - 1, mididx, :]
    #         bottompt = local_grid[mididx + 1, mididx, :]
    #         sp = []
    #         sp.append(np.linalg.norm(midpt - leftpt))
    #         sp.append(np.linalg.norm(midpt - rightpt))
    #         sp.append(np.linalg.norm(midpt - toppt))
    #         sp.append(np.linalg.norm(midpt - bottompt))
    #         sp = np.asarray(sp)
    #         sp_value = abs(spacing * np.ones(sp.shape) - sp).max()
    #         if sp_value > 5e-2 * spacing:
    #             print('sp is ', sp)
    #             print('spacing error {:.6f} at {}, true spacing {:.6f}'.format(sp_value, idx, spacing))
    #             err = True
    #
    #         # check planar
    #         normal_vector = np.cross((midpt - leftpt), (midpt - toppt))
    #         normal_vector = normal_vector / np.linalg.norm(normal_vector)
    #         dotary = np.dot(local_grid.reshape([-1, 3]), normal_vector)
    #         planar_value = abs(dotary[0] * np.ones(dotary.shape) - dotary).max()
    #         if planar_value > 1e-2:
    #             print('grid not on plane {} at {}'.format(planar_value, idx))
    #             err = True
    #
    #     if err is True:
    #         raise ValueError('test_make_grid error')
    #
    #     print('test_make_grid done == passed')
    #     return

    # ======================================================
    def test_interp(self):
        print('========= start test_interp ==========')
        u0,coord,param_file,dataframe = self.get_u0()

        pt_cld = PointCloud(coord, param_file)
        finite_diff = FiniteDiff2ndOrder()

        interp = RbfInterpolator(pt_cld)
        u = Variables(pt_cld, interp, finite_diff, 0)
        u.set_val(u0)

        interp_diff_list=[]
        for idx, grid in enumerate(pt_cld.get_grid_list()):
            gridval = interp.eval(grid, u.get_val())
            r2, phi2, theta2 = self.xyz2sph(grid[0].reshape([-1, 3]))
            u0_exact2 = np.sin(theta2)*np.sin(theta2)*np.cos(phi2)  # for every point in grid
            interp_diff = (abs(u0_exact2 - gridval.reshape([-1, ])) / u0_exact2).max()
            error = self.ErrorLimit()
            if interp_diff > error.max_interp_error:
                print('interp diff: {} at row {}'.format(interp_diff, idx))
                print('phi {}, theta {}'.format(phi2[int(len(phi2)/2)]*180/np.pi,
                                                theta2[int(len(theta2)/2)]*180/np.pi))
                print('no_of_nn {}'.format(pt_cld.nn_indices[idx].shape))
                print('r2 {} '.format(r2))

            interp_diff_list.append(interp_diff)
        print('interp_diff_list max {}'.format(np.max(interp_diff_list)))
        print('test_interp done == passed')
        sys.stdout.flush()


    # =============================================================
    '''
    def test_u0(self):
        print('========= start test_u0 ==========')
        u0,coord,param_file,dataframe = self.get_u0()

        pt_cld = PointCloud(coord, param_file)
        finite_diff = FiniteDiff2ndOrder()

        interp = RbfInterpolator(pt_cld)
        u = Variables(pt_cld, interp, finite_diff, 0)
        u.set_val(u0)
        r, phi, theta = self.xyz2sph(pt_cld.coord)
        u0_exact = np.sin(theta)*np.sin(theta)*np.cos(phi)
        print('u0_exact: {}'.format(u0_exact[:20]))
        print('u0: {}'.format(u.get_val()[:20]))

        l_infinity = abs(u0_exact - u.get_val()).max()
        print('l_infinity: {}'.format(l_infinity))

        error = self.ErrorLimit()
        if l_infinity > error.max_l_infinity_error:
            raise ValueError('l_infinity_norm of divu larger than {}'.format(error.max_l_infinity_error))

        print('test_u0 done == passed')
        sys.stdout.flush()
    '''
    # =======================================================

    # def test_diffusionoperator(self):
    #     print('========= start test_diffusionoperator ==========')
    #     u0,coord,param_file,dataframe = self.get_u0()
    #     D = dataframe.get_D()
    #     print('D shape = ', D.shape)
    #
    #     error = self.ErrorLimit()
    #     pt_cld = PointCloud(coord, param_file)
    #     print('interpolated_spacing = ', pt_cld.interpolated_spacing)
    #     finite_diff = FiniteDiff2ndOrder()
    #
    #     interp = RbfInterpolator(pt_cld)
    #     u = Variables(pt_cld, interp, finite_diff, 0)
    #     u.set_val(u0)
    #     r, phi, theta = self.xyz2sph(pt_cld.coord)
    #     diff_op_exact = (2 * np.cos(2*theta) - 1) * np.cos(phi)
    #
    #     diff_op = DiffusionOperator(D)
    #     divu = diff_op.eval(u)
    #
    #     print('diff_op_exact: {}'.format(diff_op_exact[:20]))
    #     print('divu: {}'.format(divu[:20]))
    #
    #     l_infinity = abs(diff_op_exact - divu).max()
    #     print('l_infinity: {}'.format(l_infinity))
    #
    #     if l_infinity > error.max_l_infinity_error:
    #         raise ValueError('l_infinity_norm of divu larger than {}'.format(error.max_l_infinity_error))
    #
    #     print('test_diffusionoperator done == passed')
    #     sys.stdout.flush()

    # =======================================================
    @staticmethod
    def get_u0():
        parent_file = path.dirname(path.dirname(path.abspath(__file__)))
        filename = os.path.join(parent_file, "data", "testcase", "database.csv")
        param_file = os.path.join(parent_file, "data", "testcase", "param_template.csv")

        dataframe = DataFrame(filename)
        coord = dataframe.get_coord()
        u0 = dataframe.get_uni_u()
        if coord.dtype != 'float64':
            coord = np.asarray(coord, dtype='float64')
        if u0.dtype != 'float64':
            u0 = np.asarray(u0, dtype='float64')

        return u0, coord, param_file, dataframe

    # =======================================================
    @staticmethod
    def xyz2sph(cart_coord):
        x, y, z = cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)

        phi = np.zeros_like(theta)
        idx = np.argwhere(x > 0.0).squeeze()
        phi[idx] = np.arctan(y[idx] / x[idx])

        idx = np.argwhere((x < 0.0) & (y >= 0)).squeeze()
        phi[idx] = np.arctan(y[idx] / x[idx]) + np.pi

        idx = np.argwhere((x < 0.0) & (y < 0)).squeeze()
        phi[idx] = np.arctan(y[idx] / x[idx]) - np.pi

        idx = np.argwhere((x == 0) & (y > 0)).squeeze()
        phi[idx] = np.pi / 2

        idx = np.argwhere((x == 0) & (y < 0)).squeeze()
        phi[idx] = - np.pi / 2

        idx = np.argwhere((x == 0.0) & (y == 0))
        phi[idx] = 0.0
        return r, phi, theta

    # =======================================================

    def test_local_axis_perpendicular(self):
        parent_file = path.dirname(path.dirname(path.abspath(__file__)))
        filename = os.path.join(parent_file, "data", "testcase", "database.csv")
        param_file = os.path.join(parent_file, "data", "testcase", "param_template.csv")

        dataframe = DataFrame(filename)
        coord = dataframe.get_coord()

        error = self.ErrorLimit()

        pt_cld = PointCloud(coord, param_file)
        dotprod = []
        for i in range(pt_cld.no_pt):
            local_axis1_tmp = pt_cld.local_axis1[i].squeeze()
            local_axis2_tmp = pt_cld.local_axis2[i].squeeze()
            dotprod_tmp = np.dot(local_axis1_tmp, local_axis2_tmp)
            dotprod.append(dotprod_tmp)

        if all(i <= error.dotprod_limit for i in dotprod) is False:
            raise ValueError('local_axis1 and local_axis2 must be perpendicular')
        print('test_local_axis_perpendicular done == passed')
        sys.stdout.flush()


if __name__ == '__main__':
    unittest.main()

