
class FiniteDiff2ndOrder:

    def __init__(self):
        return

    def eval(self,gridval,mid,spacing):
        gradx = (gridval[mid, mid + 1] - gridval[mid, mid - 1]) / (2 * spacing)
        grady = (gridval[mid - 1, mid] - gridval[mid + 1, mid]) / (2 * spacing)

        return gradx,grady
