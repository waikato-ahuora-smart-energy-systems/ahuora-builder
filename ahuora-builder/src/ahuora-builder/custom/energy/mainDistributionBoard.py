# Import Pyomo libraries
from stringprep import in_table_a1
from pyomo.environ import (
    Var,
    Suffix,
    units as pyunits,
    Set,
    value
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.math import smooth_min
from idaes.core.util.exceptions import ConfigurationError

# Set up logger
_log = idaeslog.getLogger(__name__)


# When using this file the name "MDB" is what is imported
@declare_process_block_class("MDB")
class MDBData(UnitModelBlockData):
    """
    Zero order power distirbution board model
    """

    # CONFIG are options for the unit model, this simple model only has the mandatory config options
    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. The Bus unit does not support dynamic
    behavior, thus this must be False.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The Bus unit does not have defined volume, thus
    this must be False.""",
        ),
    )
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "num_inlets",
        ConfigValue(
            default=False,
            domain=int,
            description="Number of inlets to add",
            doc="""Number of inlets to add""",
        ),
    )
    CONFIG.declare(
        "num_outlets",
        ConfigValue(
            default=False,
            domain=int,
            description="Number of outlets to add",
            doc="""Number of outlets to add""",
        ),
    )
    
    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()

        # This creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)


        # Defining parameters of state block class
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True  # inlet block is an inlet

        # Add state blocks for inlet, outlet, and waste
        # These include the state variables and any other properties on demand
        num_inlets = self.config.num_inlets

        self.inlet_list = [ "inlet_" + str(i+1) for i in range(num_inlets) ]
        

        self.inlet_blocks = []
        for name in self.inlet_list:
            # add properties_inlet_1, properties_inlet2 etc
            state_block = self.config.property_package.state_block_class(
                self.flowsheet().config.time, doc="inlet power", **tmp_dict
            )
            self.inlet_blocks.append(state_block)
            # Dynamic equivalent to self.properties_inlet_1 = stateblock
            setattr(self,"properties_" + name, state_block)
            # also add the port
            self.add_port(name=name,block=state_block)


        num_outlets = self.config.num_outlets
        self.outlet_list = [ "outlet_" + str(i+1) for i in range(num_outlets) ]
        self.priority = [i for i in range(num_outlets)]
        self.outlet_set = Set(initialize=self.outlet_list)


        self.outlet_blocks = []
        for name in self.outlet_list:
            # add properties_outlet_1, properties_outlet2 etc
            state_block = self.config.property_package.state_block_class(
                self.flowsheet().config.time, doc="outlet power", **tmp_dict
            )
            self.outlet_blocks.append(state_block)
            # Dynamic equivalent to self.properties_outlet_1 = stateblock
            setattr(self,"properties_" + name, state_block)
            # also add the port
            self.add_port(name=name,block=state_block)

        # Add variable for power splitting
        self.priorities= Var(
            self.flowsheet().time,
            self.outlet_set,
            initialize=1.0,
            units = pyunits.W,
            doc="How the power is split between outlets depending on priority",
        )
        #Add array to store power demand at each outlet
        self.demand = []

        self.available = Var(
            self.flowsheet().time,
            self.outlet_list,
            initialize = 1.0,
            units = pyunits.dimensionless
        )

        #Expression for total power supply:
        @self.Expression(
            self.flowsheet().time
        )
        def total_power(b, t):
            return (
                 sum(
                    state_block[t].power for state_block in self.inlet_blocks
                 )
            )
         
        #Expression for total power demand:
        @self.Expression(
            self.flowsheet().time
        )
        def total_power_demand(b, t):
            return (
                 sum(
                    state_block[t].power for state_block in self.outlet_blocks
                 )
            )

        #Constraint for available power:
        @self.Constraint(
                self.flowsheet().time,
                self.outlet_list,
                doc="expression for calculating available power"
        )
        def eq_available_power(b,t,o):       
            p = self.outlet_list.index(o)
            if o == self.outlet_list[0]:
                return self.available[t,o] == self.total_power[t]
            else:
                outlet_block = getattr(self,"properties_" + self.outlet_list[p-1])
                return self.available[t,o] == self.available[t,self.outlet_list[p-1]] - outlet_block[t].power

        
        @self.Constraint(
            self.flowsheet().time,
            self.outlet_list,
            doc = "power out"
        )   
        def eq_power_out(b,t,o):
            outlet_block = getattr(self,"properties_" + o)
            if o == self.outlet_list[-1]:
                return outlet_block[t].power == self.available[t,o]
            else:    
                return outlet_block[t].power == smooth_min(self.priorities[t,o], self.available[t,o])      

        @self.Constraint(
            self.flowsheet().time,
            doc = "Power at last outlet"
        )
        def eq_last_outlet(b,t):
            return self.priorities[t, self.outlet_list[-1]] == self.available[t,self.outlet_list[-1]]    

        
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        
    def initialize(blk, *args, **kwargs):
        for t in blk.flowsheet().time:
            power_in = 0
            for state_block in blk.inlet_blocks:
                power_in += state_block[t].power.value
            
                                   
    def _get_stream_table_contents(self, time_point=0):
        """
        Assume unit has standard configuration of 1 inlet and 1 outlet.

        Developers should overload this as appropriate.
        """
        
        io_dict = {}
        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name) # get a reference to the port
        
        io_dict = {}
        for outlet_name in self.outlet_list:
            io_dict[outlet_name] = getattr(self, outlet_name)