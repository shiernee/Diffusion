from src.variables.variables               import variables
from src.variables.linear_operator         import linear_operator
from src.variables.cubic_operator          import cubic_operator
from src.variables.diffusion_operator      import diffusion_operator
from src.variables.differential_expression import differential_expression

# solving this equation
#
#  dudt = div.(D*dudx) - u*(a-u)*(1-u) + b*w
#  dwdt = c*u + d*w
#
#

class FHNeq:

    def __init__(self,a,b,c,d,D,dt,pt_cld,interp):

        self.dt = dt

        self.deUu = differential_expression()
        self.deUw = differential_expression()
        self.deWu = differential_expression()
        self.deWw = differential_expression()

        # define the differential operators
        self.diff = diffusion_operator(D)
        self.cubic = cubic_operator(a)
        self.linear0 = linear_operator(b)
        self.linear1 = linear_operator(c)
        self.linear2 = linear_operator(d)

        # add terms into FHN equations
        self.deUu.push_back(self.diff)    #div.(D*dudw)
        self.deUu.push_back(self.cubic)   #-u*(a-u)*(1-u)
        self.deUw.push_back(self.linear0) #+b*w
        self.deWu.push_back(self.linear1) #+c*u
        self.deWw.push_back(self.linear2) #+d*w

        # define the variables to compute
        self.u0 = variables(pt_cld,interp,dt)
        self.u1 = variables(pt_cld,interp,dt)
        self.w0 = variables(pt_cld,interp,dt)
        self.w1 = variables(pt_cld,interp,dt)

        self.U = [self.u0,self.u1]
        self.W = [self.w0,self.w1]

# ==============================================
    def step(self,u_cur,w_cur,u_nxt,w_nxt):
        
        dudt  = self.deUu.eval(u_cur)
        dudt += self.deUw.eval(w_cur)
        dwdt  = self.deWu.eval(u_cur)
        dwdt += self.deWw.eval(w_cur)

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

            self.step(u_cur,w_cur,u_nxt,w_nxt)

        u.copy(u_nxt)
        w.copy(w_nxt)
        return [u,w]

# ==============================================
