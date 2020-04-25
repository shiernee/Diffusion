import numpy as np

class GenerateDC:
    def __init__(self, no_pt):
        """

        :param args: no_pt
        """

        self.D = np.zeros([no_pt, ])
        self.c = np.zeros([no_pt, ])

    def set_fixed_D(self, D):
        """

        :param D: float
        :return: void
        """
        if isinstance(D, float) is False:
            raise TypeError('Only floats are allowed')
        self.D = np.ones(self.D.shape) * D
        return


if __name__ == '__main__':
    gen_DC = GenerateDC(no_pt=1002)
    print(gen_DC.D.mean())
    print(gen_DC.c.mean())
    gen_DC.set_fixed_D(4.0)
    print(gen_DC.D.mean())
    print(gen_DC.D)
