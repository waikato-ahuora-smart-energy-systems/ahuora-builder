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

# Set up logger
_log = idaeslog.getLogger(__name__)


# When using this file the name "Load" is what is imported
@declare_process_block_class("Dsi")
class dsiData(UnitModelBlockData):
    """
    Direct Steam Injection Unit Model

    This unit model is used to represent a direct steam injection
    process. There are no degrees of freedom, but the steam is mixed with the inlet fluid to heat it up.
    It is assumed that the pressure of the fluid doesn't change, i.e the steam loses its pressure.
    However, the enthalpy of the steam remains the same.
    This allows to use two different property packages for the steam and for the inlet fluid, however,
    it only works if the reference enthalpy of the steam and the inlet fluid are the same.

    It's basically a combination of a mixer and a translator.
    """

    # CONFIG are options for the unit model
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
        "steam_property_package",
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
        "steam_property_package_args",
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
        self.properties_milk_in = self.config.property_package.state_block_class(
            self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        )

        # We need to calculate the enthalpy of the composition, before adding additional enthalpy from the temperature difference.
        # so we'll add another state block to do that.
        tmp_dict["defined_state"] = False
        tmp_dict["has_phase_equilibrium"] = False
        self.properties_mixed_unheated = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of mixture, before accounting for temperature difference",
            **tmp_dict,
        )

        # Add outlet block
        tmp_dict["defined_state"] = False
        tmp_dict["has_phase_equilibrium"] = False
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of outlet",
            **tmp_dict,
        )

        # Add steam inlet block
        steam_dict = dict(**self.config.steam_property_package_args)
        steam_dict["parameters"] = self.config.steam_property_package
        steam_dict["defined_state"] = True
        tmp_dict["has_phase_equilibrium"] = True

        self.properties_steam_in = self.config.steam_property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of steam inlet",
            **steam_dict,
        )

        # To calculate the amount of enthalpy to add to the inlet fluid, we need to know the difference in enthalpy between steam at that T and P
        # and steam at its inlet conditions. Note this is assuming that effects of composition (the steam will no longer be pure water) are negligible.
        # Note that this state block is just for calcuating, and not an actual inlet or outlet.

        steam_dict["defined_state"] = False  # This doesn't affect pure components.
        steam_dict["has_phase_equilibrium"] = True
        self.properties_steam_cooled = (
            self.config.steam_property_package.state_block_class(
                self.flowsheet().config.time,
                doc="Material properties of cooled steam",
                **steam_dict,
            )
        )

        # Add ports
        self.add_port(name="outlet", block=self.properties_out)
        self.add_port(name="inlet", block=self.properties_milk_in, doc="Inlet port")
        self.add_port(
            name="steam_inlet", block=self.properties_steam_in, doc="Steam inlet port"
        )

        # CONDITIONS

        # STEAM INTERMEDIATE BLOCK

        # Temperature (= other inlet temperature)
        @self.Constraint(
            self.flowsheet().time,
            doc="Set the temperature of the cooled steam to be the same as the inlet fluid",
        )
        def eq_steam_cooled_temperature(b, t):
            return (
                b.properties_steam_cooled[t].temperature
                == b.properties_milk_in[t].temperature
            )

        # Pressure (= other inlet pressure)
        @self.Constraint(
            self.flowsheet().time,
            doc="Set the pressure of the cooled steam to be the same as the inlet fluid",
        )
        def eq_steam_cooled_pressure(b, t):
            return (
                b.properties_steam_cooled[t].pressure
                == b.properties_milk_in[t].pressure
            )

        # Flow = steam_flow
        @self.Constraint(
            self.flowsheet().time,
            self.config.steam_property_package.component_list,
            doc="Set the composition of the cooled steam to be the same as the steam inlet",
        )
        def eq_steam_cooled_composition(b, t, c):
            return 0 == sum(
                b.properties_steam_cooled[t].get_material_flow_terms(p, c)
                - b.properties_steam_in[t].get_material_flow_terms(p, c)
                for p in b.properties_steam_in[t].phase_list
            )

        # CALCULATE ENTHALPY DIFFERENCE
        @self.Expression(
            self.flowsheet().time,
        )
        def steam_delta_h(b, t):
            """
            Calculate the difference in enthalpy between the steam inlet and the cooled steam.
            This is used to calculate the amount of enthalpy to add to the inlet fluid.
            """
            return (
                b.properties_steam_in[t].enth_mol
                - b.properties_steam_cooled[t].enth_mol
            ) * b.properties_steam_in[t].flow_mol

        # MIXING (without changing temperature)

        # Pressure (= inlet pressure)
        @self.Constraint(
            self.flowsheet().time,
            doc="Equivalent pressure balance",
        )
        def eq_mixed_pressure(b, t):
            return (
                b.properties_mixed_unheated[t].pressure
                == b.properties_milk_in[t].pressure
            )

        # Temperature (= inlet temperature)
        @self.Constraint(
            self.flowsheet().time,
            doc="Equivalent temperature balance",
        )
        def eq_mixed_temperature(b, t):
            return (
                b.properties_mixed_unheated[t].temperature
                == b.properties_milk_in[t].temperature
            )

        # Flow = inlet flow + steam flow
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.component_list,
            doc="Mass balance",
        )
        def eq_mixed_composition(b, t, c):
            return 0 == sum(
                b.properties_milk_in[t].get_material_flow_terms(p, c)
                + (
                    b.properties_steam_in[t].get_material_flow_terms(p, c)
                    if c
                    in b.properties_steam_in[
                        t
                    ].component_list  # handle the case where a component isn't in the steam inlet (e.g no milk in helmholtz)
                    else 0
                )
                - b.properties_mixed_unheated[t].get_material_flow_terms(p, c)
                for p in b.properties_milk_in[t].phase_list
                if (p, c) in b.properties_milk_in[t].phase_component_set
            )  # handle the case where a component is not in that phase (e.g no milk vapor)

        # OUTLET BLOCK

        # Pressure (= inlet pressure)
        @self.Constraint(
            self.flowsheet().time,
            doc="Pressure balance",
        )
        def eq_outlet_pressure(b, t):
            return b.properties_out[t].pressure == b.properties_milk_in[t].pressure

        # Enthalpy (= mixed enthalpy + delta steam enthalpy)
        @self.Constraint(
            self.flowsheet().time,
            doc="Energy balance",
        )
        def eq_outlet_combined_enthalpy(b, t):
            return b.properties_out[t].enth_mol == b.properties_mixed_unheated[
                t
            ].enth_mol + (b.steam_delta_h[t] / b.properties_mixed_unheated[t].flow_mol)
        
        # Flow = mixed flow

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.component_list,
            doc="Mass balance for the outlet",
        )
        def eq_outlet_composition(b, t, c):
            return 0 == sum(
                b.properties_out[t].get_material_flow_terms(p, c)
                - b.properties_mixed_unheated[t].get_material_flow_terms(p, c)
                for p in b.properties_out[t].phase_list
                if (p, c) in b.properties_out[t].phase_component_set
            )  # handle the case where a component is not in that phase (e.g no milk vapor)



    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

    def initialize(blk, *args, **kwargs):
        blk.properties_milk_in.initialize()
        blk.properties_steam_in.initialize()

        for t in blk.flowsheet().time:
            # copy temperature and pressure from properties_milk_in to properties_steam_cooled
            # blk.properties_steam_cooled[t].temperature.set_value(
            #     blk.properties_milk_in[t].temperature.value
            # )
            blk.properties_steam_cooled[t].pressure.set_value(
                blk.properties_milk_in[t].pressure.value
            )
            # Copy composition from properties_steam_in to properties_steam_cooled
            blk.properties_steam_cooled[t].flow_mol.set_value(
                blk.properties_steam_in[t].flow_mol.value
            )
            # If it's steam, there's only one component, so we prolly don't need to worry about composition.
            # But may want TODO this for other cases.

        blk.properties_steam_cooled.initialize()
        blk.properties_mixed_unheated.initialize()

        blk.properties_out.initialize()
        pass

    def _get_stream_table_contents(self, time_point=0):
        """
        Assume unit has standard configuration of 1 inlet and 1 outlet.

        Developers should overload this as appropriate.
        """
        try:
            return create_stream_table_dataframe(
                {
                    "outlet": self.outlet,
                    "inlet": self.inlet,
                    "steam_inlet": self.steam_inlet,
                },
                time_point=time_point,
            )
        except AttributeError:
            raise ConfigurationError(
                f"Unit model {self.name} does not have the standard Port "
                f"names (inlet and outlet). Please contact the unit model "
                f"developer to develop a unit specific stream table."
            )
