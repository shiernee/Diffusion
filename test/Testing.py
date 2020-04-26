import unittest
import numpy as np
from src.utils.Parameters import Parameters
from src.utils.RbfInterpolator import RbfInterpolator
from src.pointcloud.PointCloud import PointCloud
from src.variables.Sum import Sum
from src.variables.LinearOperator import LinearOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.BaseVariables import BaseVariables
from src.variables.Variables import Variables



class Testing(unittest.TestCase):

    # example to learn
    def test_sum(self):
        sum = Sum(4, 5)
        self.assertEqual(sum.eval(), 9)
        print('test_sum done')

    # =======================================
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
        case = 'testcase1'
        pt_cld = PointCloud(case)
        param = Parameters(case)
        interp = RbfInterpolator(pt_cld)
        u = Variables(pt_cld, interp, 0)
        u.set_val(param.u0)

        r, phi, theta = self.xyz2sph(pt_cld.coord)
        diff_op_exact = 2 * np.cos(2*theta - 1) * np.cos(phi)
        print(diff_op_exact)

        diff_op = DiffusionOperator(1)
        divu = diff_op.eval(u)
        print(divu)

        l_infinity = abs(diff_op_exact - divu).max()
        print(l_infinity)

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


if __name__ == '__main__':
    unittest.main()