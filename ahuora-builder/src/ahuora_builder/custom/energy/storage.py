# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Suffix,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core.util.exceptions import ConfigurationError

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from pyomo.util.check_units import assert_units_consistent
from pyomo.environ import value

# Set up logger
_log = idaeslog.getLogger(__name__)


# When using this file the name "Link" is what is imported
@declare_process_block_class("Storage")
class StorageData(UnitModelBlockData):
    """
    Zero order Link model
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
        tmp_dict["defined_state"] = True  
        self.properties_in = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )


        # Add ports - oftentimes users interact with these rather than the state blocks
        self.add_port(name="inlet", block=self.properties_in)
        self.add_port(name="outlet", block=self.properties_out)

        # Add variables:
        self.charging_efficiency = Var(
            self.flowsheet().config.time,
            initialize=1.0,
            doc="Charging Efficiency",
            )
        self.capacity = Var(self.flowsheet().config.time,
            initialize=1.0,
            units=pyunits.kWh,
            doc="Capacity of the storage",
            )
        self.initial_SOC = Var(self.flowsheet().config.time,
            initialize=0.4,
            doc="Initial State of Charge",
            )
        self.charging_power_in = Var(self.flowsheet().config.time,
            initialize=1.0,
            units=pyunits.W,
            doc="Power in",
            )
        self.charging_power_out = Var(self.flowsheet().config.time,
            initialize=1.0,
            units=pyunits.W,
            doc="Power out",
            )

        @self.Expression(
            self.flowsheet().time,
            doc="updated state of charge",
        )
        def updated_SOC(b,t):
            power_in = self.charging_power_in[t]
            power_out = self.charging_power_out[t]
            power_change = (power_in-power_out)/(pyunits.W *1000)
            capacity_without_unit = self.capacity[t]/pyunits.kWh
            if t == self.flowsheet().time.first():
                self.updated_SOC[t] = self.initial_SOC[t] + power_change/capacity_without_unit
            else:
                self.updated_SOC[t] = self.updated_SOC[t-1] + power_change/capacity_without_unit
          
            return self.updated_SOC[t]



        @self.Constraint(
            self.flowsheet().time,
            doc="Set output power for charging",
        )
        

        @self.Constraint(
            self.flowsheet().time,
            doc="Set output power for discharging",
        )
        def set_power_out_discharge(b,t): 
            return b.properties_out[t].power == self.charging_power_out[t]
       

        @self.Constraint(
            self.flowsheet().time,
            doc="Set output power for charging",
        )
        def set_power_charge(b,t):
            return self.charging_power_in[t] == self.properties_in[t].power

        
        # Add a constraint to ensure power_change is within a range
        @self.Constraint(
            self.flowsheet().time,
            doc="Ensure power_change is within a specified range",
        )
        def power_change_within_range(b, t):
            power_in = self.charging_power_in[t]
            power_out = self.charging_power_out[t]
            power_in_out = (power_in-power_out)/(pyunits.W *1000)
           # power_in_out = b.properties_out[t].power / (pyunits.W * 1000)
            remaining_power = power_in_out + self.initial_SOC[t] * self.capacity[t]
            return remaining_power <= self.capacity[t]
        
        @self.Constraint(
            self.flowsheet().time,
            doc="Ensure power_change is within capacity",
        )
        def power_change_above_zero(b, t):
            
            power_in = self.charging_power_in[t]
            power_out = self.charging_power_out[t]
            power_in_out = (power_in-power_out)/(pyunits.W *1000)
            #power_in_out = b.properties_out[t].power / (pyunits.W * 1000)
            remaining_power = power_in_out + self.initial_SOC[t] * self.capacity[t]
            return remaining_power >= 0
    
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
    
    def initialize(blk, *args, **kwargs):

        for i in blk.properties_in.index_set():
            if blk.initial_SOC[i].value == 0 and abs(blk.charging_power_out[i].value)> 0 and abs(blk.charging_power_in[i].value) == 0:
                raise ConfigurationError(
                    "Warning: There is no power to discharge in the battery."
                )
            if blk.charging_power_in[i].value > blk.capacity[i].value:
                pass
                #    raise ConfigurationError(
                #     "Warning: Battery capacity is exceeded!"
                # )       

    def _get_stream_table_contents(self, time_point=0):
        pass