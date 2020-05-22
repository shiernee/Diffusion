from src.variables.Variables import Variables
from src.variables.LinearOperator import LinearOperator
from src.variables.ConstantOperator import ConstantOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.ConstantExpression import ConstantExpression
from src.variables.DifferentialExpression import DifferentialExpression
from src.variables.DiffusionOperatorFitMethod import DiffusionOperatorFitMethod
import numpy as np


# solving this equation
#
#  dudt = div.(D*dudx) - u*(a-u)*(1-u) + b*w
#  dwdt = c*u + d*w
#
#  dudt = div.(D*dudx) + u*(a-u)*(u-1) + w  #SN
#  dwdt = epsilon*(beta*u - gamma*w - delta) #SN
#


class FHNeq:

    def __init__(self, a, b, epsilon_beta, neg_epsilon_gamma, neg_epsilon_delta, D,
                 pt_cld, interp, dt):
        self.dt = dt

        self.deUu = DifferentialExpression()  # du/dt = f1(u)
        self.deUw = DifferentialExpression()  # du/dt = f2(w)
        self.deWu = DifferentialExpression()  # dw/dt = f3(u)
        self.deWw = DifferentialExpression()  # dw/dt = f4(w)
        self.deWc = ConstantExpression()  # dw/dt = f5()

        # define the differential operators
        # dudt = div.(D*dudx) + u*(a-u)*(U-1) + b*w
        self.diff = DiffusionOperatorFitMethod(D, pt_cld, interp)
        self.cubic = CubicOperator(a)
        self.linear0 = LinearOperator(b)
        # add in
        self.deUu.push_back(self.diff)  # div.(D*dudw)
        self.deUu.push_back(self.cubic)  # -u*(a-u)*(1-u)
        self.deUw.push_back(self.linear0)  # +b*w

        # dwdt = epsilon*beta*u - epsilon*gamma*w - epsilon*delta
        self.linear1 = LinearOperator(epsilon_beta)
        self.linear2 = LinearOperator(neg_epsilon_gamma)
        self.constant = ConstantOperator(neg_epsilon_delta)
        # add in
        self.deWu.push_back(self.linear1)  # epsilon*beta*u
        self.deWw.push_back(self.linear2)  # -epsilon*gamma*w
        self.deWc.push_back(self.constant)  # -epsilon*delta

        # define the variables to compute
        self.u0 = Variables(pt_cld, interp, self.dt)
        self.u1 = Variables(pt_cld, interp, self.dt)
        self.w0 = Variables(pt_cld, interp, self.dt)
        self.w1 = Variables(pt_cld, interp, self.dt)

        self.U = [self.u0, self.u1]
        self.W = [self.w0, self.w1]

    # ==============================================
    def _step(self, u_cur, w_cur, idx_nxt, u_nxt, w_nxt):
        dudt = self.deUu.eval(u_cur)
        dudt += self.deUw.eval(w_cur)

        dwdt = self.deWu.eval(u_cur)
        dwdt += self.deWw.eval(w_cur)
        dwdt += self.deWc.eval()

        u_cur.eval(dudt, self.dt)
        w_cur.eval(dwdt, self.dt)

        self.U[idx_nxt].copy(u_cur)
        self.W[idx_nxt].copy(w_cur)

    # ==============================================
    def integrate(self, u, w, nsteps):
        self.u0.copy(u)  # initialize the variables
        self.u1.copy(u)
        self.w0.copy(w)
        self.w1.copy(w)

        for itr in range(nsteps):
            print('itr: {}/{}'.format(itr, nsteps))
            idx_cur = (itr) % 2
            idx_nxt = (itr + 1) % 2
            u_cur = self.U[idx_cur]
            u_nxt = self.U[idx_nxt]
            w_cur = self.W[idx_cur]
            w_nxt = self.W[idx_nxt]

            self._step(u_cur, w_cur, idx_nxt, u_nxt, w_nxt)

        u.copy(u_nxt)
        w.copy(w_nxt)
        return [u, w]

# ==============================================
