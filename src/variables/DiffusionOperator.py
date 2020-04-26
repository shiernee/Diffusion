

class DiffusionOperator:

    def __init__(self,D):
        self.D = D # diffusion coefficient

    def eval(self,u):

        u.eval_ddx()
        dudx = u.get_ddx()

        dudx.multiply(self.D) # D must be shape of dudx
        dudx.eval_div()
        divu = dudx.get_div()
        return divu


