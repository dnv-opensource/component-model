from pythonfmu import Fmi2Causality, Fmi2Initial, Fmi2Variability, Fmi2Slave, DefaultExperiment # for work with PythonFMU
import uuid

class WrapFMU(Fmi2Slave):
    '''Fmi2Slave implementation of a model made in Python, performing the FMU 'packaging', implements the pythonfmu.Fmi2Slave and runs buildFMU, i.e.
          * prepare the modeldescription.xml
          * implement the FMI2 C interface for the present platform (as .dll, .so, ...)  

       The following is expected of any valid Python model:
          * a complete list of variables including meta information, the global Variable.variableList contains that

    Args:
       model (obj): reference to the (instantiated) model object

       licenseTxt (str)='Open Source'
       copyright (str)='See copyright notice of used tools'
       defaultExperiment (dict) = None: key/value dictionary for the default experiment setup
       guid (str)=None: Unique identifier of the model (supplied or automatically generated)
       non_default_flags (dict)={}: Any of the defined FMI flags with a non-default value (see FMI 2.0.4, Section 4.3.1)
       **kwargs: Any other keyword argument (transferred to the basic slave object)
    
    '''
    def __init__(self, model,
                 licenseTxt:str='Open Source', copyrightTxt:str='See copyright notice of used tools',
                 defaultExperiment:dict=None, nonDefaultFlags:dict=None, guid=None, **kwargs):
        self.model = model
        self.default_experiment = DefaultExperiment( None, None, None, None) if defaultExperiment is None else DefaultExperiment( **defaultExperiment)
        print("EXP", self.default_experiment)
        kwargs.update( { 'instance_name':__name__, 'modelName':self.model.name, 'description':self.model.description,
                         'author':self.model.author, 'version':self.model.version, 'license':licenseTxt, 'copyright':copyrightTxt,
                         'guid':guid if guid is not None else uuid.uuid4().hex, 
                         'default_experiment':self.default_experiment})
        super().__init__( **kwargs)
        print("FLAGS", nonDefaultFlags)
        self.nonDefaultFlags = self.check_flags( nonDefaultFlags)
        return
        self.enter_initialization_mode() # ensure that the variables get correct values
#         self.register_variable(Real( name="y0", causality=Fmi2Causality.parameter, description="y position at time 0", initial=Fmi2Initial.exact, variability=Fmi2Variability.fixed))
#         self.register_variable(Real( name="angle0", causality=Fmi2Causality.parameter, description="angle at time 0", initial=Fmi2Initial.exact, variability=Fmi2Variability.fixed))
#         self.register_variable(Real( name="v0", causality=Fmi2Causality.parameter, description="speed at time 0", initial=Fmi2Initial.exact, variability=Fmi2Variability.fixed))
#         self.register_variable(Real( name="bounceFactor", causality=Fmi2Causality.parameter, description="factor on speed when bouncing", initial=Fmi2Initial.exact, variability=Fmi2Variability.fixed))
#         self.register_variable(Real( name="drag", causality=Fmi2Causality.parameter, description="drag decelleration factor defined as a = self.drag* v^2 with dimensin 1/m", initial=Fmi2Initial.exact, variability=Fmi2Variability.fixed))
#         self.register_variable(Real( name="x", causality=Fmi2Causality.output, description="x position at time", initial=Fmi2Initial.exact, variability=Fmi2Variability.continuous))
#         self.register_variable(Real( name="y", causality=Fmi2Causality.output, description="y position at time", initial=Fmi2Initial.exact, variability=Fmi2Variability.continuous))
#         self.register_variable(Real( name="v_x", causality=Fmi2Causality.output, description="speed in x-direction at time", initial=Fmi2Initial.exact, variability=Fmi2Variability.continuous))
#         self.register_variable(Real( name="v_y", causality=Fmi2Causality.output, description="speed in y-direction at time", initial=Fmi2Initial.exact, variability=Fmi2Variability.continuous))
#         self.register_variable(String("mdShort", causality=Fmi2Causality.local))
        
        # Note:
        # it is also possible to explicitly define getters and setters as lambdas in case the variable is not backed by a Python field.
        # self.register_variable(Real("myReal", causality=Fmi2Causality.output, getter=lambda: self.realOut, setter=lambda v: set_real_out(v))

    def build(self):
        scriptFile = self.instance_name+'.py'
        with tempfile.TemporaryDirectory() as documentation_dir:
            doc_dir = Path(documentation_dir)
            license_file = doc_dir / "licenses" / "license.txt"
            license_file.parent.mkdir()
            license_file.write_text("Dummy license")
            index_file = doc_dir / "index.html"
            index_file.write_text("dummy index")
            asBuilt = FmuBuilder.build_FMU( scriptFile, dest='.', documentation_folder=doc_dir)#, xFunc=None)
            
    def enter_initialization_mode(self):
        a0 = radians( self.angle0)
        self.x = 0.0 # start always at x=0
        self.y = 1.0*self.y0
        self.v_x = self.v0* cos( a0)
        self.v_y = self.v0* sin( a0)
        self.energy = 9.81*self.y + 0.5*self.v_y*self.v_y
        self.period = 2* self.v_y/ 9.81 # may change when energy is taken out of the system
        print("INIT y0:", self.y0,", angle:", self.angle0, "v_x_0:", self.v_x, ", v_y_0:", self.v_y, ", bounce:", self.bounceFactor, ", drag:", self.drag)
        return( True)
        
    def do_step(self, current_time, step_size):
        def bounce_loss( v0):
            if self.bounceFactor == 1.0:
                return( v0)
            v0 *= self.bounceFactor # speed with which it leaves the ground
            self.energy = v0*v0/2
            self.period = 2*v0/9.81
            return( v0)
        
        print("STEP (", current_time, '). [',self.x, ',',self.y,']', 'v: [', self.v_x,',',self.v_y,']', sep="")
        self.x += self.v_x* step_size
        y = self.y + self.v_y* step_size - 9.81/ 2* step_size*step_size
        if y <= 0 and self.v_y < 0: # bounce
            t0 = self.v_y/9.81* (1 - sqrt( 1 + 2*self.y*9.81/ self.v_y/ self.v_y)) # time when it hits the ground
            v0 = sqrt(2*self.energy) #more exact than self.v_y - 9.81* t0 # speed when jumps off the ground (without energy loss)
            v0 = bounce_loss( v0) # check energy loss during bouncing
            #print("BOUNCE", current_time, '(', self.x, ',', self.y,')', t0, v0)
            tRest = step_size-t0
            while True:
                if tRest < self.period: # cannot do a whole bounce in the remaining time
                    break
                if self.drag != 0: raise NotImplementedError("Bouncing a whole period is not implemented when drag is involved. Try choosing smaller time steps.")
                v0 = bounce_loss( v0)
                tRest -= self.period
                
            self.y = v0* tRest  - 9.81/ 2* tRest*tRest # height end of step
            self.v_y = v0 - 9.81* tRest # speed end of step
        else:
            self.v_y -= 9.81* step_size
            self.y = y
            print("V_y, y", self.v_y, self.y)
        if self.drag != 0:
            fac = 1 - self.drag* sqrt( self.v_x*self.v_x + self.v_y*self.v_y)* step_size
            self.v_x *= fac
            self.v_y *= fac
            self.energy = 9.81*self.y + 0.5*self.v_y*self.v_y
            #print("FAC", fac, self.v_x, self.v_y, self.energy)
        e = 9.81*self.y + 0.5*self.v_y*self.v_y
        if  abs( e-self.energy) > 1e-6: # and  and
            print("Energy leak", current_time, e, self.energy)
            self.energy = e
        print(current_time, self.x, self.y, self.v_x, self.v_y)
        return True
    
    @staticmethod
    def check_flags( flags):
        def check_flag( fl, typ):
            if fl in flags:
                if isinstance( flags[fl], bool) and flags[fl]: # a nondefault value
                    _flags.update( { fl: flags[fl]})
                elif isinstance( flags[fl], int) and flags[fl]!=0: # nondefault integer
                    _flags.update( { fl: flags[fl]})
        _flags = {}
        check_flag( 'needsExecutionTool', bool)
        check_flag( 'canHandleVariableCommunicationStepSize', bool)
        check_flag( 'canInterpolateInputs', bool)
        check_flag( 'maxOutputDerivativeOrder', int)
        check_flag( 'canRunAsynchchronously', bool)
        check_flag( 'canBeInstantiatedOnlyOncePerProcess', bool)
        check_flag( 'canNotUstMemoryManagementFunctions', bool)
        check_flag( 'canGetAndSetFMUstate', bool)
        check_flag( 'canSerializeFMUstate', bool)
        check_flag( 'providesDirectionalDerivative', bool)
        return( _flags)
    
    def to_xml(self, el):
        ''' Add information to the provided etree element, as needed, and return '''
        if el.tag == 'CoSimulation': #ToDo: may need to check whether also other information than the flags need to be included
            el.attrib.update( self.nonDefaultFlags)