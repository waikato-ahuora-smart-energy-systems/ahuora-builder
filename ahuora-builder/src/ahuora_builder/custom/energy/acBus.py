# Import Pyomo libraries
from stringprep import in_table_a1
from pyomo.environ import (
    Constraint,
    Set,
    Var,
    Suffix,
    units as pyunits,
)
from pyomo.environ import Reals
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core.util.model_statistics import degrees_of_freedom
# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    MaterialBalanceType,
    MaterialFlowBasis,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.models.unit_models import separator
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError, BurntToast, PropertyNotSupportedError

from idaes.models.unit_models import separator
#Import enum
from enum import Enum

# Set up logger
_log = idaeslog.getLogger(__name__)

# Enumerate options for balances
class SplittingType(Enum):
    """
    Enum of supported material split types.
    """

    totalFlow = 1
    phaseFlow = 2
    componentFlow = 3
    phaseComponentFlow = 4



# When using this file the name "acBus" is what is imported
@declare_process_block_class("acBus")
class acBusData(UnitModelBlockData):
    """
    Zero order acbus model
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
            doc="Number of inlets to add",
        ),
    )
    CONFIG.declare(
        "num_outlets",
        ConfigValue(
            default=False,
            domain=int,
            description="Number of outlets to add",
            doc="Number of outlets to add",
        ),
    )
    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
            **default** - MaterialBalanceType.useDefault.
            **Valid values:** {
            **MaterialBalanceType.useDefault - refer to property package for default
            balance type
            **MaterialBalanceType.none** - exclude material balances,
            **MaterialBalanceType.componentPhase** - use phase component balances,
            **MaterialBalanceType.componentTotal** - use total component balances,
            **MaterialBalanceType.elementTotal** - use total element balances,
            **MaterialBalanceType.total** - use total material balance.}""",
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
        self.inlet_set = Set(initialize=self.inlet_list) 
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


        # Add outlet state blocks

        num_outlets = self.config.num_outlets
        self.outlet_list = [ "outlet_" + str(i+1) for i in range(num_outlets) ]
        self.outlet_set = Set(initialize=self.outlet_list)
        self.outlet_blocks = []

        for name in self.outlet_list:
            tmp_dict["defined_state"] = False
            state_block = self.config.property_package.state_block_class(
                self.flowsheet().config.time, doc="outlet power", **tmp_dict
            )
            self.outlet_blocks.append(state_block)
            setattr(self,"properties_" + name, state_block)
            self.add_port(name=name,block=state_block)
 
        # Add variable for power splitting
        self.split_fraction = Var(
            self.flowsheet().time,
            self.outlet_set,
            initialize=1.0,
            #units = pyunits.dimensionless,
            doc="How the power is split between outlets",
        )
       
        #Obtain the power components from the inlets
        @self.Expression(
            self.flowsheet().time,
        )
        def total_active_power(b,t):
            return  sum(state_block[t].active_power for state_block in self.inlet_blocks)
        
        @self.Expression(
            self.flowsheet().time,
        )
        def total_reactive_power(b,t):
            return  sum(state_block[t].reactive_power for state_block in self.inlet_blocks)
        
        @self.Expression(
                self.flowsheet().time,
        )
        def total_voltage(b,t):
            return  sum(state_block[t].voltage for state_block in self.inlet_blocks)

        #Add constraints
    
        @self.Constraint(
            self.flowsheet().time,
            self.outlet_list,
            doc="active power split",   
        )
        def eq_active_power(b,t,o):
            outlet_block = getattr(self,"properties_" + o)
            return outlet_block[t].active_power == (
                self.total_active_power[t] * b.split_fraction[t,o]) 

        @self.Constraint(
            self.flowsheet().time,
            self.outlet_list,
            doc="reactive power split", 
        )
        def eq_reactive_power(b,t,o):
            outlet_block = getattr(self,"properties_" + o)
            return outlet_block[t].reactive_power == (
                self.total_reactive_power[t] * b.split_fraction[t,o])
       
        @self.Constraint(
            self.flowsheet().time,
            self.outlet_list,
            doc="voltage split", 
        )
        def eq_voltage(b,t,o):
            outlet_block = getattr(self,"properties_" + o)
            return outlet_block[t].voltage == (
                self.total_voltage[t] * b.split_fraction[t,o])    


        @self.Constraint(
            self.flowsheet().time,
            doc="Split fraction sum to 1",
        )
        def eq_split_fraction_sum(b, t):
            return sum(b.split_fraction[t, o] for o in self.outlet_list) == 1.0     
        
    
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
    
    def initialize(blk, *args, **kwargs):
        
       pass


    def _get_stream_table_contents(self, time_point=0):

        io_dict = {}
        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name) # get a reference to the port
        
        out_dict = {}
        for outlet_name in self.outlet_list:
            out_dict[outlet_name] = getattr(self, outlet_name) # get a reference to the port


   
