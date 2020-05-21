
# can use scipy and follow this code
# scipy.linalg.lstsq
# https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

import scipy
import numpy as np


class QuadraticFormDivCalculator:

    def __init__(self):
        return

    # =====================================================
    # fit a linear form
    # D = d[0] + d[1]*x + d[2]*y
    # then return c
    # values of x,y,u are contain in grid2D, and val
    # grid2D[0] is the first point with 
    # for a 3x3 grid (2D), there are 9 points
    # *note* center point on grid is always (xc,yc) = (0,0)
    def fit_linear(self, grid2D, val):

        #prepare the A by putitng in formula
        A = np.c_[np.ones(grid2D.shape[0]), grid2D[:, 0], grid2D[:, 1]]
        d,_,_,_ = scipy.linalg.lstsq(A, val)
        return d # d is the cofficient of fit

    # =====================================================
    # fit a quadratic form
    # u = c[0] + c[1]*x + c[2]*y + c[3]*x*y + c[4]*x*x + c[5]*y*y
    # then return c
    # values of x,y,u are contain in grid2D, and val
    # grid2D[0] is the first point with 
    # for a 3x3 grid (2D), there are 9 points
    # *note* center point on grid is always (xc,yc) = (0,0)

    def fit_quadratic(self,grid2D,val):
        #prepare the A by putitng in formula
        A = np.c_[np.ones(grid2D.shape[0]), grid2D[:,:2], np.prod(grid2D[:,:2], axis=1), grid2D[:,:2]**2]
        c,_,_,_ = scipy.linalg.lstsq(A, val)
        return c # c is the cofficient of fit

        # =====================================================
        # fit a cubic form
        # u = c[0] + c[1]*x + c[2]*y + c[3]*x*y + c[4]*x*x + c[5]*y*y + c[6]*x*x*y + c[7]*x*y*y + c[8]*x*x*x +
        # c[9]*y*y*y
        # then return c
        # values of x,y,u are contain in grid2D, and val
        # grid2D[0] is the first point with
        # for a 3x3 grid (2D), there are 9 points
        # *note* center point on grid is always (xc,yc) = (0,0)

    def fit_cubic(self, grid2D, val):
        # prepare the A by putitng in formula
        A = np.c_[np.ones(grid2D.shape[0]), grid2D[:, :2], np.prod(grid2D[:, :2], axis=1), grid2D[:, :2] ** 2, \
            grid2D[:, 0]*grid2D[:, 1] ** 2 + grid2D[:, 0] ** 2 * grid2D[:, 1] + grid2D[:, 0]**3 + grid2D[:, 1]**3]
        c, _, _, _ = scipy.linalg.lstsq(A, val)
        return c  # c is the cofficient of fit

    # =====================================================
    # grid3D is the grid in 3D
    # interp_u is the interpolated u values on grid
    # interp_D is the interpolated D values on grid
    # spacing is spacing on grid (interpolated spacing)
    def eval(self, grid3D, interp_u, interp_D, spacing):
        # use grid3D to generate control points for 
        # u and for D
        x = np.arange(0, len(grid3D)) - len(grid3D)//2
        y = np.arange(0, len(grid3D)) - len(grid3D)//2
        XX, YY = np.meshgrid(x, y)
        grid2D = YY.reshape([-1, ]), XX.reshape([-1, ])
        grid2D = np.asarray(grid2D).T
        grid2D = grid2D * spacing

        val_u = interp_u.reshape([-1, ])
        val_D = interp_D.reshape([-1, ])

        c = self.fit_quadratic(grid2D, val_u)
        d = self.fit_linear(grid2D, val_D)

        return d[1]*c[1] + d[2]*c[2] + 2*d[0]*(c[4]+c[5])       

