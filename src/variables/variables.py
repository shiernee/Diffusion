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

from src.variables.BaseVariables import BaseVariables
from src.variables.GradVariables import GradVariables


class Variables(BaseVariables):
    # use point cloud to specify the
    # storage and differentials of variable

    def __init__(self,point_cloud,interpolator,gradient_algo,t):
        super(Variables,self).__init__(t)

        self.ddx_updated = False
        self.point_cloud = point_cloud
        self.interpolator = interpolator

        self.gradient_algo = gradient_algo

        self.grad = GradVariables(point_cloud,interpolator,gradient_algo,t)

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
        for grid in self.point_cloud.get_grid_list():
            # grid gives the control points and points to be interpolated
            # control points means the points in point cloud with function
            # values. control points include the current point and neighbor
            # points
            gridval = self.interpolator.eval(grid, self.val)
            mid = int(len(gridval)/2)

            gradx,grady = self.gradient_algo.eval(gridval,mid,self.point_cloud.interpolated_spacing)
            ddx.append([gradx, grady])
        ddx = np.asarray(ddx)  
        self.check_bounds(ddx, 'Variables.py58') # always check

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
            
