# Import Pyomo libraries
from stringprep import in_table_a1
from pyomo.environ import (
    Var,
    Suffix,
    units as pyunits,
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

from idaes.core.util.exceptions import ConfigurationError, BurntToast

# Set up logger
_log = idaeslog.getLogger(__name__)


# When using this file the name "EnergyMixer" is what is imported
@declare_process_block_class("EnergyMixer")
class EnergyMixerData(UnitModelBlockData):
    """
    Zero order energy_mixer model
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


        # Add outlet state block
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )

        # Add outlet port
        self.add_port(name="outlet", block=self.properties_out)

        #Add variables for capacity and efficiency:
        self.efficiency = Var(self.flowsheet().config.time,
            initialize=1.0, 
            doc="Efficiency of the link",
        )
        self.capacity = Var(
            initialize=1.0,
            units = pyunits.W,
            doc="Capacity of the link",
        )
        

        # Add constraints
        # Usually unit models use a control volume to do the mass, energy, and momentum
        # balances, however, they will be explicitly written out in this example
        
        @self.Constraint(
            self.flowsheet().time,
            doc="Power usage",
        )
        def eq_power_balance(b, t):
            return (
                 sum(
                    state_block[t].power for state_block in self.inlet_blocks
                 ) * self.efficiency[t]
                  == b.properties_out[t].power
            )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
    
    def initialize(blk, *args, **kwargs):
        # Just propagate the power from inlet to outlet, good simple method of initialization
        for t in blk.flowsheet().time:
            power_in = 0
            for state_block in blk.inlet_blocks:
                power_in += state_block[t].power.value
            if not blk.properties_out[t].power.fixed:
                blk.properties_out[t].power = power_in
            if(power_in > blk.capacity.value):
                raise BurntToast(
                    "Danger: Input power exceeds energy mixer capacity. Please either increase capacity or lower input power.".format(blk.name)
                )
    def _get_stream_table_contents(self, time_point=0):
        """
        Assume unit has standard configuration of 1 inlet and 1 outlet.

        Developers should overload this as appropriate.
        """
        
        io_dict = {}
        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name) # get a reference to the port
        
        io_dict["Outlet"] = self.outlet
        return create_stream_table_dataframe(io_dict, time_point=time_point)