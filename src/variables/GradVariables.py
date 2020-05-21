import copy as cp
import numpy as np

from src.variables.BaseVariables import BaseVariables


class GradVariables(BaseVariables):

    # HK def __init__(self,point_cloud,interpolator,t):
    def __init__(self,point_cloud,interpolator,gradient_algo,t):
        super(GradVariables,self).__init__(t)

        #print('GradVaiables.py interpolator ',interpolator)

        self.gradient_algo = gradient_algo
        self.div_updated = False
        self.val_copy    = None 
        self.div_copy    = None
        self.div         = None

        self.interpolator = interpolator
        self.point_cloud  = point_cloud

    # =====================================
    def reset_interpolator(self,interpolator):
        self.interpolator = interpolator
    # =====================================
    # compute the divergence of gradient
    # =====================================
    def eval_div(self): # divergence
        self.div_updated = True
        self.div = []
        for grid in self.point_cloud.get_grid_list():
            gridvalx = self.interpolator.eval(grid, self.val[:, 0])
            gridvaly = self.interpolator.eval(grid, self.val[:, 1])
            mid = int(len(gridvalx) / 2)

            gradxx,gradxy = self.gradient_algo.eval(gridvalx,mid,self.point_cloud.interpolated_spacing)
            gradyx,gradyy = self.gradient_algo.eval(gridvaly,mid,self.point_cloud.interpolated_spacing)
            #gradx = (gridvalx[mid, mid + 1] - gridvalx[mid, mid - 1]) / (2 * self.point_cloud.interpolated_spacing)
            #grady = (gridvaly[mid - 1, mid] - gridvaly[mid + 1, mid]) / (2 * self.point_cloud.interpolated_spacing)
            #print('gradxx ',gradxx)
            #print('gradxy ',gradxy)
            #print('gradyx ',gradyx)
            #print('gradyy ',gradyy)
            #print('div spacing ',self.point_cloud.interpolated_spacing)
            #print('gridvalx ',gridvalx)
            #print('gridvaly ',gridvaly)

            self.div.append(gradxx + gradyy)

        self.div = np.asarray(self.div)
        self.check_bounds(self.div,'GradVariables.py53')
    # =====================================
    # set the gradient values
    # =====================================
    # def set_val(self,val):
    #     self.val = np.copy(val)
    # =====================================
    # return the whole numpy array of ddx
    # w.r.t. local coordinate
    # return shape = [n,1] - vector
    # =====================================

    def get_div(self):
        assert self.div_updated is True, 'eval before query'
        self.div_copy = np.copy(self.div)
        return self.div_copy
    # =====================================
    # overload multiplication operator with numpy
    # =====================================
    def multiply(self,D):
        D = D.get_val()
        if np.ndim(D) == 1:
            D = D.reshape([-1, 1])
        self.val = self.val*D

   
