# Import Pyomo libraries
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
from idaes.core.util.exceptions import ConfigurationError
# Set up logger
_log = idaeslog.getLogger(__name__)



# When using this file the name "Transformer" is what is imported
@declare_process_block_class("Transformer")
class TransformerData(UnitModelBlockData):
    """
    Zero order Transformer model
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
    **default** - False. The Transformer unit does not have defined volume, thus
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

    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()

        # This creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)


        # Add state blocks for inlet, outlet, and waste
        # These include the state variables and any other properties on demand
        # Add inlet block
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True  # inlet block is an inlet
        self.properties_in = self.config.property_package.state_block_class(
            self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        )
        # Add outlet and waste block
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )

        # Add ports - oftentimes users interact with these rather than the state blocks
        self.add_port(name="inlet", block=self.properties_in)
        self.add_port(name="outlet", block=self.properties_out)

        # # Add variable for turns_ratio
        # self.turns_ratio = Var(self.flowsheet().config.time,
        #     initialize=1.0,
        #     doc="Turns Ratio of the Transformer",
        # )
        # # Add variable for Load Resistance
        # self.resistance = Var(self.flowsheet().config.time,
        #     initialize=1.0,
        #     doc="Load Resistance of the Transformer",
        # )
        self.n_capacity = Var(
            self.flowsheet().config.time,
            initialize=1.0,
            units = pyunits.W,
            doc="N Capacity of the Transformer",
        )
        self.voltage = Var(self.flowsheet().config.time,
            initialize=1.0,
            units = pyunits.V,
            doc="Voltage Capacity of the Transformer",
        )
        self.efficiency = Var(self.flowsheet().config.time,
            initialize=1.0,
            units = pyunits.dimensionless,
            doc="Efficiency of the Transformer",
        )


        # Add constraints
        @self.Constraint(
            self.flowsheet().time,
            doc = "Power out",
        )
        def power_out(b, t):
            return (
                self.properties_out[t].power
                == self.properties_in[t].power * self.efficiency[t]
            )
        
        @self.Constraint(
            self.flowsheet().time,
            doc = "Capacity check",
        )
        def capacity_check(b,t):
            return abs(self.properties_in[t].power) <= self.n_capacity[t]

        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Voltage out",
        # )
        # def voltage_out(b, t):
        #     return (
        #         b.properties_in[t].voltage * self.turns_ratio[t] == self.properties_out[t].voltage
        #     )
        # @self.Constraint(
        #     self.flowsheet().time,
        #     doc="Current out",
        # )
        # def current_out(b,t):
        #     return(
        #         self.properties_out[t].voltage / self.resistance[t] == self.properties_out[t].current
        #     )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
    
    def initialize(blk, *args, **kwargs):
        # Just propagate the power from inlet to outlet, good simple method of initialization
        for i in blk.properties_in.index_set():
            if not blk.properties_out[i].power.fixed:
                blk.properties_out[i].power = blk.properties_in[i].power.value
            
            if abs(blk.properties_in[i].power.value) > blk.n_capacity[i].value:
                 raise ConfigurationError(
                    "Danger: Input power exceeds transformer capacity. Please either increase capacity or lower input power."
                )