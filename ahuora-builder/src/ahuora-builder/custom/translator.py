# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Suffix,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError
from idaes.models.unit_models.translator import TranslatorData
# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from enum import Enum

# Set up logger
_log = idaeslog.getLogger(__name__)

class TranslatorType(Enum):
    """
    Enum of supported translator types. This allows you to change what variables are translated across.
    """
    pressure_enthalpy = "pressure_enthalpy"
    pressure_vapor_fraction = "pressure_vapor_fraction"
    pressure_temperature = "pressure_temperature"


# When using this file the name "GenericTranslator" is what is imported
@declare_process_block_class("GenericTranslator")
class GenericTranslatorData(TranslatorData):
    """
    GenericTranslator.

    This is used to translate between two different property packages, and supports dropping compounds that are not present.

    For example, if you have a stream of water/milk, and it's almost all water, this allows you to translate the stream to a water-only stream.

    It works by fixing the temperature and pressure, and flow of each component in the outlet stream to be the same as the inlet stream.
     
    """
    CONFIG = TranslatorData.CONFIG()

    CONFIG.declare(
        "translator_type",
        ConfigValue(
            default=TranslatorType.pressure_enthalpy.value,
            description="Translator type to use for translating properties",
            doc="""
            Depending on the property packages you are using and the phases that are present, it may make sense to use different translators.
            This allows you to select what variables are translated across.
            """,
        ),
    )


    def build(self):
        self.CONFIG.outlet_state_defined = False # See constraint for flow
        #self.CONFIG.has_phase_equilibrium = True # I don't think it matters if this is set, becuase in theory the phase equilibrium should
        # already have been calculated in the inlet stream.
        super().build()

        # Pressure (= inlet pressure)
        @self.Constraint(
            self.flowsheet().time,
            doc="Pressure balance",
        )
        def eq_outlet_pressure(b, t):
            return b.properties_in[t].pressure == b.properties_out[t].pressure

        # Flow
        @self.Constraint(
            self.flowsheet().time,
            self.config.outlet_property_package.component_list,
            doc="Mass balance for the outlet",
        )
        def eq_outlet_composition(b, t, c):
            return 0 == sum(
                b.properties_out[t].get_material_flow_terms(p, c)
                - b.properties_in[t].get_material_flow_terms(p, c)
                for p in b.properties_out[t].phase_list
                if (p, c) in b.properties_out[t].phase_component_set
            ) 

        if self.config.translator_type == TranslatorType.pressure_enthalpy.value:
            # Enthalpy (= inlet enthalpy)
            @self.Constraint(
                self.flowsheet().time,
                doc="Enthalpy balance",
            )
            def eq_outlet_enth_mol(b, t):
                return (
                    b.properties_in[t].enth_mol == b.properties_out[t].enth_mol
                )
        elif self.config.translator_type == TranslatorType.pressure_vapor_fraction.value:
            # Vapor fraction (= inlet vapor fraction)
            # TODO: We might be able to make this smoother, by extending the vapor fraction
            # below 0 and above 1 to make solving more reliable. See
            # https://github.com/waikato-ahuora-smart-energy-systems/PropertyPackages/blob/8c6ee67b9d028ba0fdd1c937d9dcda821595b7d1/property_packages/helmholtz/helmholtz_extended.py#L104
            @self.Constraint(
                self.flowsheet().time,
                doc="Vapor fraction balance",
            )
            def eq_outlet_vapor_frac(b, t):
                return (
                    b.properties_in[t].vapor_frac == b.properties_out[t].vapor_frac
                )
        elif self.config.translator_type == TranslatorType.pressure_temperature.value:
            # Temperature (= inlet temperature)
            @self.Constraint(
                self.flowsheet().time,
                doc="Temperature balance",
            )
            def eq_outlet_temperature(b, t):
                return (
                    b.properties_in[t].temperature == b.properties_out[t].temperature
                )
        else:
            raise ConfigurationError(
                f"Translator type {self.CONFIG.translator_type} is not supported."
            )
        
        

