import copy as cp
import numpy as np


class ConstChecker:
    def __init__(self):
        self.ref_list = []
        self.cpy_list = []
        self.checked_flag = False

    def insert(self, obj):
        self.check_flag = False
        self.ref_list.append(obj)
        self.cpy_list.append(cp.copy(obj))
        return

    def check(self):
        self.checked_flag = True
        paired_list = zip(self.ref_list, self.cpy_list)
        for ref, cpy in paired_list:
            assert self.instance_check(cpy, ref), 'error:object modified'
        return

    # def check(self):
    #     self.checked_flag = True
    #     paired_list = zip(self.ref_list, self.cpy_list)
    #     for ref, cpy in paired_list:
    #         assert ref == cpy, 'error:object modified'
    #     return

    def __del__(self):
        assert self.checked_flag, 'error: checker goes out of scope without checking'
        return

    def instance_check(self, a, b):
        # specialize for numpy
        if type(a).__module__ == np.__name__:
            return np.array_equal(a, b)
        return a.__eq__(b)


class A:
    def __init__(self, v):
        self.u = v
        self

    def __eq__(self, obj):
        return obj.u == self.u

    # def __del__(self):
    #     print('call delete')


if __name__=='__main__':

    a = A(4)
    b = A(5)
    print('a.u', a.u)
    print('b.u', b.u)
    cc = ConstChecker()
    cc.insert(a)
    cc.insert(b)

    a.u = 9 # go change a
    print('a.u', a.u)
    cc.check()
