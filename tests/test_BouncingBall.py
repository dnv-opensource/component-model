import numpy as np
from math import sqrt
from component_model.model import Model
from component_model.variable import Variable, Variable_NP
from fmpy import dump, simulate_fmu, plot_result


class BouncingBallFMU(Model):
    '''Fmi2Slave implementation of a model made in Python, performing the FMU 'packaging', implements the pythonfmu.Fmi2Slave and runs buildFMU, i.e.
          * prepare the modeldescription.xml
          * implement the FMI2 C interface for the present platform (as .dll, .so, ...)  

       The following is expected of any valid Python model:
          * a complete list of variables including meta information, the model.variables dictionary contains that
          * a do_step method specifying what happens at each simulation step

    Args:
       model (obj): reference to the (instantiated) model object

       licenseTxt (str)='Open Source'
       copyright (str)='See copyright notice of used tools'
       defaultExperiment (dict) = None: key/value dictionary for the default experiment setup
       guid (str)=None: Unique identifier of the model (supplied or automatically generated)
       non_default_flags (dict)={}: Any of the defined FMI flags with a non-default value (see FMI 2.0.4, Section 4.3.1)
       **kwargs: Any other keyword argument (transferred to the basic slave object)
    
    '''
    def __init__(self,
                 name           ="BouncingBall",
                 description    ="Simple bouncing ball test FMU",
                 author         = "DNV, SEACo project",
                 version        = "0.1",
                 **kwargs
                 ):
        super().__init__( name=name, description=description, author=author, version=version, **kwargs)
        self.x = Variable_NP( self, initialVal=(0.0,0.0), name="BallPosition", description='''Position of ball (x,z) at time.''',
                              causality='output', variability='continuous', 
                              on_step = None) #lambda t, dT: self.boom0.rotate( axis=self.craneAngularVelocity.value) if np.any(self.craneAngularVelocity.value!=0) else None,
        self.v = Variable_NP( self, initialVal=(1.0,1.0), name="v0", description="speed at time as (x,z) vector",
                              causality='output', variability='continuous')
        self.bounceFactor = Variable( self, initialVal = 0.95, name="bounceFactor", description="factor on speed when bouncing", causality='parameter', variability='fixed',)
        self.drag = Variable( self, initialVal=0.0, name="drag", description="drag decelleration factor defined as a = self.drag* v^2 with dimension 1/m", causality='parameter', variability='fixed',)
        self.energy = Variable( self, initialVal=0.0, name="energy", description="Total energy of ball in J", causality='output', variability='continuous',)
        self.period = Variable( self, initialVal=0.0, name="period", description="Bouncing period of ball", causality='output', variability='continuous',)
#        self.register_variable( String("mdShort", causality=Fmi2Causality.local))

    def enter_initialization_mode(self):
        self.energy.value = 9.81*self.x.value[1] + 0.5*np.dot( self.v.value, self.v.value)
        self.period.value = 2* self.v.value[1]/ 9.81 # may change when energy is taken out of the system
        return( True)
        
    def do_step(self, current_time, step_size):
        print("ENERGY", self.energy, type(self.energy))
        def bounce_loss( v0):
            if self.bounceFactor.value == 1.0:
                return( v0)
            v0 *= self.bounceFactor.value # speed with which it leaves the ground
            self.energy.value = v0*v0/2
            self.period.value = 2*v0/9.81
            return( v0)

        self.x.value[0] += self.v.value[0]* step_size
        y = self.x.value[1] + self.v.value[1]* step_size - 9.81/ 2* step_size**2
        if y <= 0 and self.v.value[1] < 0: # bounce
            t0 = self.v.value[1]/9.81* (1 - sqrt( 1 + 2*self.x.value[1]*9.81/ self.v.value[1]**2)) # time when it hits the ground
            v0 = sqrt(2*self.energy.value) #more exact than self.v_y - 9.81* t0 # speed when jumps off the ground (without energy loss)
            v0 = bounce_loss( v0) # check energy loss during bouncing
            #print("BOUNCE", current_time, self.x.value , t0, v0)
            tRest = step_size-t0
            while True:
                if tRest < self.period.value: # cannot do a whole bounce in the remaining time
                    break
                if self.drag.value != 0: raise NotImplementedError("Bouncing a whole period is not implemented when drag is involved. Try choosing smaller time steps.")
                v0 = bounce_loss( v0)
                tRest -= self.period.value
                
            self.x.value[1] = v0* tRest  - 9.81/ 2* tRest*tRest # height end of step
            self.v.value[1] = v0 - 9.81* tRest # speed end of step
        else:
            self.v.value[1] -= 9.81* step_size
            self.x.value[1] = y
        if self.drag != 0:
            fac = 1 - self.drag.value* np.linalg.norm( self.v.value)* step_size
            self.v.value *= fac
            self.energy.value = 9.81*self.x.value[0] + 0.5*np.dot( self.v.value, self.v.value)
            #print("FAC", fac, self.v.value, self.energy)
#         e = 9.81*self.x.value[1] + 0.5*self.v.value[1]**2
#         if  abs( e-self.energy) > 1e-6: # and  and
#             print("Energy leak", current_time, e, self.energy)
#             self.energy = e
        return True


if __name__ == '__main__':
    testIt = 1
    asBuilt = Model.build( 'test_BouncingBall.py')
    if testIt>0:
        result = simulate_fmu( asBuilt.name, start_time=0.0, stop_time=10.0, step_size=0.1, solver='Euler', debug_logging=True, )
#                               start_values={ 'v.0':1.0, 'v.1':1.0, })
        print( dump( asBuilt.name))
        plot_result(result)
    elif testIt==2: # checking the dll/so
        bb = WinDLL(os.path.abspath( os.path.curdir) +"\\BouncingBall.dll")
        bb.fmi2GetTypesPlatform.restype = c_char_p
        print( bb.fmi2GetTypesPlatform(None))
        bb.fmi2GetVersion.restype = c_char_p
        print( bb.fmi2GetVersion(None))
