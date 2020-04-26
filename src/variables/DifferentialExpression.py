# ===============================
class DifferentialExpression:

    def __init__(self):
        self.terms = []
        return

    # add differential operator 
    # terms into list
    def push_back(self,term):
        self.terms.append(term)

    # evaluate the variable u using
    # list of differential operators
    def eval(self, u):
        t = []
        for term in self.terms:
            t.append(term.eval(u))
        return sum(t)

# ===============================


