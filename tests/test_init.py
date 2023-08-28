from abc import ABC

class A(ABC):
    def __init__(self, **kwargs):
        #print("KWARGS", kwargs)
        #self.__dict__.update( kwargs)
        print("DICT A", self.__dict__)

class B(A):
    def __init__(self, hi='hi', ho='ho'):
        super().__init__(hi='hi', ho='ho')
    

#A( hi='hi', ho='ho')
B()