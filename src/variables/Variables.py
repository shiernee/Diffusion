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


class Variables(BaseVariables):
    # use point cloud to specify the
    # storage and differentials of variable

    def __init__(self, point_cloud, interpolator, t):
        super(Variables, self).__init__(t)

        self.point_cloud = point_cloud
        self.interpolator = interpolator

