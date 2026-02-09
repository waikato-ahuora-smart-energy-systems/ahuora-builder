# Import Python libraries
import logging
from typing import Any

# Import Pyomo units
from pyomo.environ import units as pyunits

# Import IDAES cores
from idaes.core import LiquidPhase, VaporPhase, Component

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


# Set up logger
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration dictionary for an ideal Benzene-Toluene system

# Data Sources:
# [1] The Properties of Gases and Liquids (1987)
#     4th edition, Chemical Engineering Series - Robert C. Reid
# [3] Engineering Toolbox, https://www.engineeringtoolbox.com
#     Retrieved 1st December, 2019

def parse_value_to_float(value: str) -> float:
    """Convert value to float"""
    try:
        return float(value)
    except (TypeError, ValueError):
        _log.error(f"Could not convert {value} to float.")
        raise ValueError(f"Could not convert {value} to float.")


def get_and_parse_value_to_float(prop_map: dict[str, Any], key: str) -> float:
    """Get and convert value to float"""
    return parse_value_to_float(str(prop_map.get(key)))


def convert_compounds_properties_to_dict(compounds_properties: list[dict[str, Any]]) -> dict:
    """
    Create compounds dict for components, one custom property package may contain more than one compound
    """

    components_dict = {}

    for item in compounds_properties:
        compound = item["compound"]
        properties = item["properties"]
        prop_map = {f["propertyName"]: f["value"] for f in properties}

        value = {
            "type": Component,
            "enth_mol_ig_comp": RPP4,
            "entr_mol_ig_comp": RPP4,
            "pressure_sat_comp": RPP4,
            "phase_equilibrium_form": {("Vap", "Liq"): log_fugacity},
            "parameter_data": {
                # [1]
                "mw": (get_and_parse_value_to_float(prop_map, "mw"), pyunits.kg / pyunits.mol),
                # [1]
                "pressure_crit": (get_and_parse_value_to_float(prop_map, "pressure_crit"), pyunits.Pa),
                # [1]
                "temperature_crit": (get_and_parse_value_to_float(prop_map, "temperature_crit"),  pyunits.K),
                # [1]
                "omega": get_and_parse_value_to_float(prop_map, "omega"),
                "cp_mol_ig_comp_coeff": {   # [1]
                    "A": (get_and_parse_value_to_float(prop_map, "cp_mol_ig_comp_coeff.A"), pyunits.J / pyunits.mol / pyunits.K),
                    "B": (get_and_parse_value_to_float(prop_map, "cp_mol_ig_comp_coeff.B"), pyunits.J / pyunits.mol / pyunits.K**2),
                    "C": (get_and_parse_value_to_float(prop_map, "cp_mol_ig_comp_coeff.C"), pyunits.J / pyunits.mol / pyunits.K**3),
                    "D": (get_and_parse_value_to_float(prop_map, "cp_mol_ig_comp_coeff.D"), pyunits.J / pyunits.mol / pyunits.K**4),
                },
                # [3]
                "enth_mol_form_vap_comp_ref": (get_and_parse_value_to_float(prop_map, "enth_mol_form_vap_comp_ref"),  pyunits.J / pyunits.mol),
                "entr_mol_form_vap_comp_ref": (
                    get_and_parse_value_to_float(
                        prop_map, "entr_mol_form_vap_comp_ref"),
                    pyunits.J / pyunits.mol / pyunits.K,
                ),  # [3]
                "pressure_sat_comp_coeff": {
                    # [1]
                    "A": (get_and_parse_value_to_float(prop_map, "pressure_sat_comp_coeff.A"), None),
                    "B": (get_and_parse_value_to_float(prop_map, "pressure_sat_comp_coeff.B"), None),
                    "C": (get_and_parse_value_to_float(prop_map, "pressure_sat_comp_coeff.C"), None),
                    "D": (get_and_parse_value_to_float(prop_map, "pressure_sat_comp_coeff.D"), None),
                },
            },
        }
        components_dict[compound["name"]] = value

    return components_dict


def parse_kappa(kappa_json: dict[str, str]) -> dict[tuple[str, str], float]:
    """
    Converts JSON-style kappa dict into IDAES-compatible format.
    Example: {"toluene, benzene": "0.1"} -> {("toluene", "benzene"): 0.1}
    """
    return {
        tuple(name.strip() for name in key.split(",")): parse_value_to_float(value)
        for key, value in kappa_json.items()
    }


def parse_to_float_array(str_array: list[str]) -> list[float]:
    """Converts a list of strings to a list of floats."""
    return [parse_value_to_float(item) for item in str_array]


def encapsulate_custom_property_package(
    custom_properties: list[Any],
    compounds_properties: list[dict[str, Any]],
) -> dict:
    """
    Encapsulated the custom property package according to the structure shown below
    """

    prop_map = {f["propertyName"]: f["value"] for f in custom_properties}
    flow_mol_bounds = parse_to_float_array(
        prop_map.get("state_bounds.flow_mol"))
    temperature_bounds = parse_to_float_array(
        prop_map.get("state_bounds.temperature"))
    pressure_bounds = parse_to_float_array(
        prop_map.get("state_bounds.pressure"))

    result = {
        "components": convert_compounds_properties_to_dict(
            compounds_properties
        ),

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
            "flow_mol": (flow_mol_bounds[0], flow_mol_bounds[1], flow_mol_bounds[2], pyunits.mol / pyunits.s),
            "temperature": (temperature_bounds[0], temperature_bounds[1], temperature_bounds[2], pyunits.K),
            "pressure": (pressure_bounds[0], pressure_bounds[1], pressure_bounds[2], pyunits.Pa),
        },
        "pressure_ref": (get_and_parse_value_to_float(prop_map, "pressure_ref"), pyunits.Pa),
        "temperature_ref": (get_and_parse_value_to_float(prop_map, "temperature_ref"), pyunits.K),
        # Defining phase equilibria
        "phases_in_equilibrium": [("Vap", "Liq")],
        "phase_equilibrium_state": {("Vap", "Liq"): CubicComplementarityVLE},
        "bubble_dew_method": LogBubbleDew,
        "parameter_data": {
            "PR_kappa": parse_kappa(prop_map["PR_kappa"])
        },
    }

    return result
