

from src.variables.QuadraticFormDivCalculator import QuadraticFormDivCalculator
import numpy as np


class DiffusionOperatorFitMethod:

    def __init__(self, D, pt_cld):
        self.D = D # diffusion coefficient - this one should be Variables type
        self.pt_cld = pt_cld # use pt_cld for getting grid
        self.quadform = QuadraticFormDivCalculator()

    def eval(self, u, interpolator):

        divu = []
        for grid in self.pt_cld.get_grid_list():
            # generate interpolated points on u
            interp_u = interpolator.eval(grid, u.get_val())
            interp_D = interpolator.eval(grid, self.D.get_val())
            grid3D = grid[0]
            val = self.quadform.eval(grid3D, interp_u, interp_D, self.pt_cld.interpolated_spacing)
            divu.append(val)

        divu = np.asarray(divu) # do we need as array?

        return divu


