import numpy as np

# ================================
# dummy class
class dummy_point_cloud:
    def __init__(self):
        self.n_points = 4
        return

    def make_local_grids(self):
        return 0  # see variables.py lines 50-70
    
    def grid_list(self):
        a_dummy_var = np.ones(10) # just to make code run
        return a_dummy_var # see variables.py lines 50-70


