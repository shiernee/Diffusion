

# term u

class linear_operator:

    def __init__(self,b):
        self.b = b

    def eval(self,u):
        u0 = u.get_val()
        return (u0*self.b)


