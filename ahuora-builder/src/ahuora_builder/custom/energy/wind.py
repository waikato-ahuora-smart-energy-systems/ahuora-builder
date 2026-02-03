# Import Pyomo libraries
from ast import Expression
from cmath import pi
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

from idaes.core.util.exceptions import ConfigurationError

# Set up logger
_log = idaeslog.getLogger(__name__)


# When using this file the name "Wind" is what is imported
@declare_process_block_class("Wind")
class WindData(UnitModelBlockData):
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
        #tmp_dict["defined_state"] = True  # inlet block is an inlet
       
        # Add outlet
        tmp_dict["defined_state"] = False  # outlet is not an inlet
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )

        # Add ports - oftentimes users interact with these rather than the state blocks
       
        self.add_port(name="outlet", block=self.properties_out)

        # Add variable for efficiency
        self.efficiency = Var(
            initialize=1.0,
            doc="Efficiency of the turbine",
        )

        # Add variable for Windspeed
        self.windspeed = Var(self.flowsheet().config.time,
            initialize=1.0, units = pyunits.m/pyunits.s,
            doc="Speed of wind hitting the turbines",
        )

        # Add variable for Pressure
        self.pressure = Var(self.flowsheet().config.time,
            initialize=1.0, units = pyunits.Pa,
            doc="Air pressure",
        )

        # Add variable for Temperature
        self.air_temp = Var(self.flowsheet().config.time,
            initialize=1.0, units = pyunits.degK,
            doc="Air Temperature",
        )

        # Add variable for Humidity
        self.humidity = Var(self.flowsheet().config.time,
            initialize=1.0,
            doc="Air humidity",
        )

        # Add variable for rotor diameter
        self.rotor_diameter = Var(
            initialize=1.0, units = pyunits.m,
            doc="Rotor Diameter",
        )

        # Add variable for no. of turbines
        self.turbine_count = Var(
            initialize=1.0,
            doc="Number of turbines",
        )

        # Simple expressions
        self.radius = self.rotor_diameter / 2
        self.area = pi * self.radius**2
        self.Cp= 0.579  # Power coefficient, typically between 0.4 and 0.6 for wind turbines

        
        # Expression indexed by time
        @self.Expression(
            self.flowsheet().time
        )
        def density(b,t):
            return(
               (self.pressure[t]/(287.05 * self.air_temp[t])) * (1-0.378*(self.humidity[t]))
            )    


        # Add constraints
        @self.Constraint(
            self.flowsheet().time,
            doc="Power usage",
        )
        def eq_power_balance(b, t):
            return (
               0.5 * self.density[t] * self.area * self.windspeed[t]**3 * self.Cp * self.efficiency * self.turbine_count == b.properties_out[t].power
            )


    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
    
    def initialize(blk, *args, **kwargs):
        # Just propagate the power from inlet to outlet, good simple method of initialization
       pass

    def _get_stream_table_contents(self, time_point=0):
        """
        Assume unit has standard configuration of 1 inlet and 1 outlet.

        Developers should overload this as appropriate.
        """
        try:
            return create_stream_table_dataframe(
                {"Outlet": self.outlet}, time_point=time_point
            )
        except AttributeError:
            raise ConfigurationError(
                f"Unit model {self.name} does not have the standard Port "
                f"names (inlet and outlet). Please contact the unit model "
                f"developer to develop a unit specific stream table."
            )