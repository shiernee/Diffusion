from src.variables.Variables               import Variables
from src.variables.LinearOperator         import LinearOperator
from src.variables.CubicOperator          import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.DifferentialExpression import DifferentialExpression
from src.variables.DiffusionOperatorFitMethod import DiffusionOperatorFitMethod
import numpy as np

# solving this equation
#
#  dudt = div.(D*dudx) - u*(a-u)*(1-u) + b*w
#  dwdt = c*u + d*w
#
#  dudt = div.(D*dudx) + u*(a-u)*(U-1) + w  #SN
#  dwdt = epsilon*(beta*u - gamma*w - delta) #SN
#


class FHNeq:

    def __init__(self, a, b, c, d, D, pt_cld, interp, finite_diff):

        self.dt = 0.5 * pt_cld.interpolated_spacing ** 2 / np.max(D)

        self.deUu = DifferentialExpression()
        self.deUw = DifferentialExpression()
        self.deWu = DifferentialExpression()
        self.deWw = DifferentialExpression()
        self.deWc = ConstantExpression()

        # define the differential operators
        # self.diff = DiffusionOperator(D)
        self.diff = DiffusionOperatorFitMethod(D, pt_cld)
        self.cubic = CubicOperator(a)
        self.linear0 = LinearOperator(b)
        self.linear1 = LinearOperator(epsilon * beta)
        self.linear2 = LinearOperator(- epsilon * gamma)
        self.constant = -epsilon * delta
        #  dwdt = c*u + d*w
        #  dwdt = epsilon*(beta*u - gamma*w - delta) #SN
        # add terms into FHN equations
        self.deUu.push_back(self.diff)    #div.(D*dudw)
        self.deUu.push_back(self.cubic)   #-u*(a-u)*(1-u)
        self.deUw.push_back(self.linear0)  #+b*w
        self.deWu.push_back(self.linear1)  #epsilon*beta*u
        self.deWw.push_back(self.linear2)  #-epsilon*gamma*w
        self.deWc.push_back(self.constant)  # -epsilon*delta

        # define the variables to compute
        self.u0 = Variables(pt_cld, interp,finite_diff, self.dt)
        self.u1 = Variables(pt_cld, interp,finite_diff, self.dt)
        self.w0 = Variables(pt_cld, interp,finite_diff, self.dt)
        self.w1 = Variables(pt_cld, interp,finite_diff, self.dt)

        self.U = [self.u0,self.u1]
        self.W = [self.w0,self.w1]

    # ==============================================
    def _step(self,u_cur,w_cur,u_nxt,w_nxt):
        
        dudt  = self.deUu.eval(u_cur)
        dudt += self.deUw.eval(w_cur)
        dwdt  = self.deWu.eval(u_cur)
        dwdt += self.deWw.eval(w_cur)
        dwdt += self.deWc.eval()

        print('dudt {}'.format(dudt))

        u_nxt.eval(dudt,self.dt)
        w_nxt.eval(dwdt,self.dt)

# ==============================================
    def integrate(self,u,w,nsteps):

        self.u0.copy(u)  # initialize the variables
        self.u1.copy(u)
        self.w0.copy(w)
        self.w1.copy(w)

        for itr in range(nsteps):
            print('itr: {}/{}'.format(itr, nsteps))
            idx_cur = (itr  )%2
            idx_nxt = (itr+1)%2
            u_cur = self.U[idx_cur]
            u_nxt = self.U[idx_nxt]
            w_cur = self.W[idx_cur]
            w_nxt = self.W[idx_nxt]

            self._step(u_cur,w_cur,u_nxt,w_nxt)

        u.copy(u_nxt)
        w.copy(w_nxt)
        return [u,w]

# ==============================================
