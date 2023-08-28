class Test:
    def __init__(self, x, on_set=None):
        self._x = x
        self._on_set = on_set
        
        
    def set_x(self, newVal):
        if self._on_set is None:
            self._x = newVal
        else:
            self._x = self._on_set(newVal)
    
    @property
    def x(self):
        return(self._x)
    @x.setter
    def x(self, newVal):
        self._x = newVal
        
        
t = Test(0)
print("Initial", t._x)
t.set_x( 1)
print("set_x", t._x)

t2 = Test(0, lambda x: 2*x)
print("Initial", t2._x)
t2.set_x( 1)
print("set_x", t2._x)

