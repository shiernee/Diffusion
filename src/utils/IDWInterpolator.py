import numpy as np
from scipy.spatial.distance import cdist


class IDWInterpolator:
    """
    x_target:
    x: neighbour coordinate
    d: array of value at neighbour
    """
    def __init__(self, x, d):
        if np.ndim(x) != 2:
            raise ValueError('x must be 2D array')
        if not all([len(x) == len(d)]):
            raise ValueError("Array x and d lengths must be equal")

        self.x = np.asarray(x).reshape([-1, 3])
        self.d = np.asarray(d).reshape([1, -1])
        return

    def _call_norm(self, x1, x2):
        return cdist(x1, x2, 'euclidean')

    def __call__(self, x_target):
        if np.ndim(x_target) != 2:
            raise ValueError('x_target must be 2D array')

        r = self._call_norm(x_target, self.x) + 1e-10  #add 1e-10 to prevent r=0
        self.inv_euc_dist = (1 / r ** 2).T

        nominator = np.dot(self.d, self.inv_euc_dist)
        denomintor = np.sum(self.inv_euc_dist, axis=0)

        return (nominator / denomintor).squeeze()
