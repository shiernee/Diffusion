# ===============================
class ConstantExpression:

    def __init__(self):
        self.terms = []
        return

    # add differential operator
    # terms into list
    def push_back(self,term):
        self.terms.append(term)

    # evaluate the variable u using
    def eval(self):
        t = []
        for term in self.terms:
            t.append(term.eval())
        return sum(t)

# ===============================