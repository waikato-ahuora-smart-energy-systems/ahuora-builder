from idaes.core.base.property_base import PhysicalParameterBlock
from property_packages.modular.modular_extended import GenericExtendedParameterBlock
from ahuora_builder_types.flowsheet_schema import PropertyPackageType, CustomCompoundPropertiesSchema
from idaes.core import LiquidPhase, VaporPhase, Component
from pyomo.environ import units as pyunits
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ceos import Cubic, CubicType
from idaes.models.properties.modular_properties.phase_equil import (
    CubicComplementarityVLE,
)
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    LogBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.forms import log_fugacity
from idaes.models.properties.modular_properties.pure import RPP4






def build_custom_package(property_package_schema: PropertyPackageType) -> PhysicalParameterBlock:
    """
    Build a custom property package from a schema
    """
    if property_package_schema.custom_compound_properties is None:
        raise ValueError("Custom compound properties must be provided for custom property package")
    if property_package_schema.custom_package_properties is None:
        raise ValueError("Custom package properties must be provided for custom property package")
    if property_package_schema.custom_kappa_values is None:
        raise ValueError("Custom kappa values must be provided for custom property package")
    

    pp = property_package_schema.custom_package_properties
    # assert nothing in pp is None
    for key, value in pp.model_dump().items():
        if value is None:
            raise ValueError(f"Custom package property {key} must be provided for custom property package")

    return GenericExtendedParameterBlock(
        **{
            "components": build_components(property_package_schema.compounds,property_package_schema.custom_compound_properties),
                # Specifying phases
            "phases": {
                "Liq": {
                    "type": LiquidPhase,
                    "equation_of_state": Cubic,
                    "equation_of_state_options": {"type": CubicType.PR},
                },
                "Vap": {
                    "type": VaporPhase,
                    "equation_of_state": Cubic,
                    "equation_of_state_options": {"type": CubicType.PR},
                },
            },
            # Set base units of measurement
            "base_units": {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            },
            # Specifying state definition
            "state_definition": FTPx,
            "state_bounds": {
                "flow_mol": (pp.flow_mol_min, pp.flow_mol_nominal, pp.flow_mol_max, pyunits.mol / pyunits.s),
                "temperature": (pp.temperature_min, pp.temperature_nominal, pp.temperature_max, pyunits.K),
                "pressure": (pp.pressure_min, pp.pressure_nominal, pp.pressure_max, pyunits.Pa),
            },
            "pressure_ref": (pp.pressure_ref, pyunits.Pa),
            "temperature_ref": (pp.temperature_ref, pyunits.K),
            # Defining phase equilibria
            "phases_in_equilibrium": [("Vap", "Liq")],
            "phase_equilibrium_state": {("Vap", "Liq"): CubicComplementarityVLE},
            "bubble_dew_method": LogBubbleDew,
            "parameter_data": {
                "PR_kappa": build_kappas(property_package_schema.custom_kappa_values)
            },
        }
    )



def build_components(compounds: list[str], custom_compound_properties: list[CustomCompoundPropertiesSchema]) -> dict[str, dict]:
    compound_configuration: dict[str,dict] = {}
    for compound, properties in zip(compounds, custom_compound_properties):
        compound_configuration[compound] = {
            "type": Component,
            "enth_mol_ig_comp": RPP4,
            "entr_mol_ig_comp": RPP4,
            "pressure_sat_comp": RPP4,
            "phase_equilibrium_form": {("Vap", "Liq"): log_fugacity},
            "parameter_data": {
                "mw": (properties.mw, pyunits.kg / pyunits.mol),  # [1]
                "pressure_crit": (properties.pressure_crit, pyunits.Pa),  # [1]
                "temperature_crit": (properties.temperature_crit, pyunits.K),  # [1]
                "omega": properties.omega,  # [1]
                "cp_mol_ig_comp_coeff": {
                    "A": (properties.cp_mol_ig_comp_coeff_A, pyunits.J / pyunits.mol / pyunits.K),  # [1]
                    "B": (properties.cp_mol_ig_comp_coeff_B, pyunits.J / pyunits.mol / pyunits.K**2),
                    "C": (properties.cp_mol_ig_comp_coeff_C, pyunits.J / pyunits.mol / pyunits.K**3),
                    "D": (properties.cp_mol_ig_comp_coeff_D, pyunits.J / pyunits.mol / pyunits.K**4),
                },
                "enth_mol_form_vap_comp_ref": (properties.enth_mol_form_vap_comp_ref, pyunits.J / pyunits.mol),  # [3]
                "entr_mol_form_vap_comp_ref": (properties.entr_mol_form_vap_comp_ref, pyunits.J / pyunits.mol / pyunits.K),  # [3]
                "pressure_sat_comp_coeff": {
                    "A": (properties.pressure_sat_comp_coeff_A, None),  # [1]
                    "B": (properties.pressure_sat_comp_coeff_B, None),
                    "C": (properties.pressure_sat_comp_coeff_C, None),
                    "D": (properties.pressure_sat_comp_coeff_D, None),
                },
            },
        }
    return compound_configuration


def build_kappas(kappa_values: dict[str,dict[str,float]]) -> dict[tuple[str,str], float]:
    kappa_configuration: dict[tuple[str,str], float] = {}
    for comp1, values in kappa_values.items():
        for comp2, value in values.items():
            kappa_configuration[(comp1, comp2)] = value
    return kappa_configuration