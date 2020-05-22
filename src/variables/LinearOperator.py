

# term u

class LinearOperator:

    def __init__(self, b):
        self.b = b.get_val()

    def eval(self, u):
        u0 = u.get_val()
        return (u0 * self.b)


