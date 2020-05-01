from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import unittest
import numpy as np
from src.utils.RbfInterpolator import RbfInterpolator
from src.pointcloud.PointCloud import PointCloud
from src.variables.LinearOperator import LinearOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.BaseVariables import BaseVariables
from src.variables.Variables import Variables
from src.utils.DataFrame import DataFrame


class Testing(unittest.TestCase):

    class ErrorLimit:
        def __init__(self):
            self.max_l_infinity_error = 10
            self.dotprod_limit = 1e-8

    def test_linearoperator(self):
        u = BaseVariables(t=0)
        u.set_val(10)

        lin_op = LinearOperator(4)
        self.assertEqual(lin_op.eval(u), 40)
        print('test_linearoperator done')

    def test_cubicoperator(self):
        u = BaseVariables(t=0)
        u.set_val(10)

        cub_op = CubicOperator(3)
        self.assertEqual(cub_op.eval(u), 630)
        print('test_cubicoperator done')

    def test_diffusionoperator(self):
        filename = "C:\\Users\sawsn\Desktop\Shiernee\Diffusion\data\\testcase\\database.csv"
        param_file = "C:\\Users\sawsn\Desktop\Shiernee\Diffusion\data\\testcase\param_template.csv"
        dataframe = DataFrame(filename)
        coord = dataframe.get_coord()
        u0 = dataframe.get_uni_u()
        D = dataframe.get_D()
        print(D.shape)

        error = self.ErrorLimit()
        pt_cld = PointCloud(coord, param_file)

        interp = RbfInterpolator(pt_cld)
        u = Variables(pt_cld, interp, 0)
        u.set_val(u0)
        r, phi, theta = self.xyz2sph(pt_cld.coord)
        diff_op_exact = 2 * np.cos(2*theta - 1) * np.cos(phi)
        print('diff_op_exact: {}'.format(diff_op_exact[:20]))

        diff_op = DiffusionOperator(D)
        divu = diff_op.eval(u)
        print('divu: {}'.format(divu[:20]))

        l_infinity = abs(diff_op_exact - divu).max()
        print('l_infinity: {}'.format(l_infinity))

        if l_infinity > error.max_l_infinity_error:
            raise ValueError('l_infinity_norm of divu larger than {}'.format(error.max_l_infinity_error))

        print('test_diffusionoperator done')


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

    def test_local_axis_perpendicular(self):
        filename = "C:\\Users\sawsn\Desktop\Shiernee\Diffusion\data\\testcase\\database.csv"
        param_file = "C:\\Users\sawsn\Desktop\Shiernee\Diffusion\data\\testcase\param_template.csv"
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
        print('test_local_axis_perpendicular done')


if __name__ == '__main__':
    unittest.main()