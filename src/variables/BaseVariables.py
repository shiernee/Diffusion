# a class to contain variables as a function of x
# only, evaluated at a time t
# can be V or W in the FHN equation
# or can be U in diffusion equation
#

# this class provides methods and storage
# to handle variables and their differentials
# with respect to time and space

import numpy as np
import copy  as cp

class BaseVariables:

    # use point cloud to specify the
    # storage and differentials of variable
    def __init__(self, t):
        self.val = None
        self.t = t # time
        self.max_value = 1000  # HK
   # =====================================
    def eval(self,dudt,dt):
        self.ddx_updated = False
        self.ddt_updated = False
        self.val = self.val + dt * dudt
        self.t = self.t + dt
    # =====================================
    # copy variables
    # =====================================
    def copy(self,u):
        self.ddx_updated = False
        self.ddt_updated = False
        self.val = np.copy(u.val)
        self.t = cp.copy(u.t)
    # =====================================
    # return the whole numpy array of position
    # for speed computation. 
    # return shape = [n,1] - scalar function
    # =====================================
    def get_val(self):
        self.val_copy = np.copy(self.val)
        return self.val_copy
    # =====================================
    # set the value
    # =====================================
    def set_val(self,numpyval):
        self.ddx_updated = False
        self.ddt_updated = False
        self.val = np.copy(numpyval)
    # =====================================
    # return the whole numpy array of ddt
    # w.r.t. local coordinate
    # return shape = [n,1] - scalar
    # =====================================
    #def get_ddt(self):
    #    assert self.ddt_updated is True, 'please eval before query'
    #    self.ddt_copy = np.copy(self.ddt)
    #    return self.ddt_copy
    # =====================================
    # HK: check that ddx contains valid numbers
    # =====================================
    def check_bounds(self,values):
        bool = np.logical_not(abs(values) < self.max_value)
        assert not(np.any(bool)),'variables.py: invalid numeric in ddx'

