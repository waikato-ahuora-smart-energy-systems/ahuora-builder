# Import Pyomo libraries
from stringprep import in_table_a1
from pyomo import environ
from pyomo.environ import (
    Constraint,
    Set,
    Var,
    Suffix,
    units as pyunits,
    atan
)
from pyomo.environ import Reals
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.contrib.fbbt import interval
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

import math
# Enumerate options for balances
class SplittingType(Enum):
    """
    Enum of supported material split types.
    """

    totalFlow = 1
    phaseFlow = 2
    componentFlow = 3
    phaseComponentFlow = 4



# When using this file the name "transmissionLine" is what is imported
@declare_process_block_class("transmissionLine")
class transmissionLineData(UnitModelBlockData):
    """
    Zero order transmission line model
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

        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["parameters"] = self.config.property_package

                # Add state blocks for inlet, outlet, and waste
        # These include the state variables and any other properties on demand
        # Add inlet block
        # tmp_dict = dict(**self.config.property_package_args)
        # tmp_dict["parameters"] = self.config.property_package
        # tmp_dict["defined_state"] = True  # inlet block is an inlet
        # self.properties_in = self.config.property_package.state_block_class(
        #     self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        # )
       
        # Add outlet and waste block
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )
         # Add second outlet block
        tmp_dict["defined_state"] = False 
        self.properties_out2 = self.config.property_package.state_block_class(
            self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        )

        # Add ports - oftentimes users interact with these rather than the state blocks
        # self.add_port(name="inlet_1", block=self.properties_in)
        # self.add_port(name="inlet_2", block=self.properties_in2)
        self.add_port(name="outlet_1", block=self.properties_out)
        self.add_port(name="outlet_2", block=self.properties_out2)       
       
        self.power_transfer = Var(self.flowsheet().config.time,
            initialize=1.0,
            doc="Power transferred between busses",
            units = pyunits.W

        )

        # Add constraints:
        @self.Constraint(
            self.flowsheet().time,
            doc = "Power balance"
        )
        def eq_power_balance(b,t):
            return self.properties_out2[t].power == self.power_transfer[t]
        
        @self.Constraint(
            self.flowsheet().time,
            doc = "Power calculation"
        )
        def eq_power_calc(b,t):
            return self.properties_out[t].power == self.power_transfer[t] * -1


                
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
    
    def initialize(blk, *args, **kwargs):
       for i in blk.properties_out.index_set():
            if not blk.properties_out[i].power.fixed:
                blk.properties_out[i].power = blk.power_transfer[i].value   


    def _get_stream_table_contents(self, time_point=0):
        """
        Assume unit has standard configuration of 1 inlet and 1 outlet.

        Developers should overload this as appropriate.
        """
       

   
