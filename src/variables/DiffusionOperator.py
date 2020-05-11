

class DiffusionOperator:

    def __init__(self, D):
        self.D = D # diffusion coefficient

    def eval(self, u, interpolator=None):
        """

        :rtype: object
        """
        u.eval_ddx()
        dudx = u.get_ddx()
        if interpolator is not None:
            dudx.reset_interpolator(interpolator)
        dudx.multiply(self.D) # D must be shape of dudx
        dudx.eval_div()
        divu = dudx.get_div()
        return divu


