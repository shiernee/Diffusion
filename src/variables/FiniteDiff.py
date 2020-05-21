
class FiniteDiff2ndOrder:

    def __init__(self):
        return

    def eval(self,gridval,mid,spacing):
        gradx = (gridval[mid, mid + 1] - gridval[mid, mid - 1]) / (2 * spacing)
        grady = (gridval[mid - 1, mid] - gridval[mid + 1, mid]) / (2 * spacing)

        return gradx,grady


class FiniteDiff4thOrder:

    def __init__(self):
        return

    def eval(self, gridval, mid, spacing):

        gradx = 1/12*gridval[mid, mid-2] - 2/3*gridval[mid, mid-1] + \
                2/3*gridval[mid, mid+1] - 1/12*gridval[mid, mid+2]
        grady = 1/12*gridval[mid-2, mid] - 2/3*gridval[mid-1, mid] + \
                2/3*gridval[mid+1, mid] - 1/12*gridval[mid+1, mid]

        gradx = gradx / (2 * spacing)
        grady = grady / (2 * spacing)

        return gradx,grady
