# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Suffix,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core.util.tables import create_stream_table_dataframe
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

from ..custom.updated_pressure_changer import (

    CompressorData,
)




# When using this file the name "CustomCompressor" is what is imported
@declare_process_block_class("CustomCompressor")
class CustomCompressorData(CompressorData):
    """
    Zero order Load model
    """

    # CONFIG are options for the unit model, this simple model only has the mandatory config options
    CONFIG = CompressorData.CONFIG()

    CONFIG.declare(
        "power_property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for power",
            doc="""Power Property parameter object used to define power calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "power_property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing power property packages",
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
 
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True  # inlet block is an inlet
       # Add inlet block
        # self.properties_in = self.config.property_package.state_block_class(
        #     self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        # )


        # Add outlet and waste block
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet
        self.power_properties_out = self.config.power_property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict
        )

        # Add ports - oftentimes users interact with these rather than the state blocks
        self.add_port(name="power_outlet", block=self.power_properties_out)

        # Add constraints
        # Usually unit models use a control volume to do the mass, energy, and momentum
        # balances, however, they will be explicitly written out in this example
        @self.Constraint(
            self.flowsheet().time,
            doc="Power out",
        )
        def eq_power_out(b, t):
            return (
                self.power_properties_out[t].power == self.work_mechanical[t] * -1
            )
    