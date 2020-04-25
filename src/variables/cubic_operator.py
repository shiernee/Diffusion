import numpy as np

# term u*(a-u)*(u-1)

class cubic_operator:

    def __init__(self,a):
        self.a = a # diffusion coefficient

    def eval(self,u):

        u0 = u.get_val()
        au = self.a - u0
        u1 = u0 - np.ones(u0.shape)
        all = -1*u0*(au)*(u1)
        return all


