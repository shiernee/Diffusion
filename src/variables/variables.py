# a class to contain variables as a function of x
# only, evaluated at a time t
# can be V or W in the FHN equation
# or can be U in diffusion equation
#

# this class provides methods and storage
# to handle variables and their differentials
# with respect to time and space

import numpy as np
import copy 

from src.variables.base_variables import base_variables
from src.variables.grad_variables import grad_variables

class variables(base_variables):

    # use point cloud to specify the
    # storage and differentials of variable
    def __init__(self,point_cloud,interpolator,t):
        super(variables,self).__init__(t)

        self.ddx_updated = False
        self.point_cloud = point_cloud
        self.interpolator = interpolator

        self.grad = grad_variables(point_cloud,interpolator,t)

    # =====================================
    # up is the value of u evaluated at another time
    # =====================================
    #def eval_ddt(self,up):
    #    self.ddt_updated = True
    #    dt = t - up.t # backward diff convention
    #    self.ddt = (self.val - up.val)/dt
    # =====================================
    # use point cloud to evaluate ddx
    # =====================================
    def eval_ddx(self):
        self.ddx_updated = True

        ddx = []  # HK
        for grid in self.point_cloud.grid_list():
            # grid gives the control points and points to be interpolated
            # control points means the points in point cloud with function
            # values. control points include the current point and neighbor
            # points
            gridval = self.interpolator.eval(grid, self.val)
            mid = int(len(gridval)/2)
            gradx = (gridval[mid, mid + 1] - gridval[mid, mid - 1]) / \
                    (2 * self.point_cloud.interpolated_spacing)
            grady = (gridval[mid - 1, mid] - gridval[mid + 1, mid]) / \
                    (2 * self.point_cloud.interpolated_spacing)
            ddx.append([gradx, grady])
        # SSN: do we need to convert ddx as list into numpy?
        ddx = np.asarray(ddx)  # HK
        self.check_bounds(ddx) # always check

        self.grad.set_val(ddx)
    # =====================================
    # return the whole numpy array of ddx
    # w.r.t. local coordinate
    # return shape = [n,2] - vector
    # =====================================
    def get_ddx(self):
        assert self.ddx_updated is True, 'please eval before query'
        self.ddx_updated = False
        return self.grad
            
