

class ConstantOperator:

    def __init__(self, delta):
        self.delta = delta.get_val()

    def eval(self):
        return self.delta


