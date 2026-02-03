from typing import Any, cast, NewType
from pyomo.environ import value
from pyomo.core.base.units_container import units, _PyomoUnit
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression, ExpressionData
from pyomo.core.expr import ExpressionBase, NPV_ProductExpression
from pint import UnitRegistry

ValueWithUnits = NewType("ValueWithUnits", NPV_ProductExpression)


def _get_pint_unit(unit: str) -> Any:
    """
    Get the pint unit object
    """
    pint_unit = getattr(units.pint_registry, unit, None)
    if pint_unit is None:
        raise AttributeError(f"Unit `{unit}` not found.")
    return pint_unit


def get_unit(unit: str) -> _PyomoUnit:
    """
    Get the pint unit object
    @unit: str unit type
    @return: unit object
    """
    return _PyomoUnit(_get_pint_unit(unit), units.pint_registry)


def attach_unit(value: float, unit: str | None) -> ValueWithUnits:
    """
    Attach the unit to the value
    @value: float value
    @unit: str unit
    @return: value with unit attached
    """
    value, unit = idaes_specific_convert(
        value, unit
    )  # make sure the value is in a unit that idaes supports
    if unit is None:
        return value
    pyomo_unit = get_unit(unit)
    return value * pyomo_unit


def check_units_equivalent(unit1: _PyomoUnit, unit2: _PyomoUnit) -> bool:
    """
    Check if two units are equivalent
    """
    if unit1 is None:
        unit1 = units.dimensionless
    if unit2 is None:
        unit2 = units.dimensionless
    return (
        unit1._get_pint_unit().dimensionality
        == unit2._get_pint_unit().dimensionality
    )



def idaes_specific_convert(value: float, unit: str | None) -> tuple[float, str | None]:
    """
    Convert the value to a unit that is specific to idaes
    (ie. idaes only supports K for temperature)
    @value: float value
    @unit: str unit
    @return: tuple:
        - (float) converted value
        - (str) new unit
    """
    if unit in ["degC", "degR", "degF"]:  # temperature units
        from_quantity = units.pint_registry.Quantity(value, unit)
        to_unit = units.pint_registry.K
        # can probably do the conversion/attachment in one step but haven't figured out how yet
        converted_value = from_quantity.to(to_unit)  # pint.Quantity object
        return converted_value.magnitude, "K"
    if unit in ["percent"]:  # dimensionless units
        from_quantity = units.pint_registry.Quantity(value, unit)
        to_unit = units.pint_registry.dimensionless
        converted_value = from_quantity.to(to_unit)
        return converted_value.magnitude, None
    if unit in [None, ""]:
        return value, None
    return value, unit


def get_attached_unit(
    var: Var | Expression | ExpressionBase | ExpressionData | float,
) -> _PyomoUnit | None:
    """
    Get the unit of a variable.
    """
    if isinstance(var, float):
        # no attached unit
        return units.dimensionless
    if isinstance(var, ExpressionData):
        var = var.parent_component()
    if var.is_indexed():
        # we will get the unit from the first item in the indexed variable
        var = var[next(iter(var.index_set()))]
    if isinstance(var, (ExpressionBase, Expression, ExpressionData)):
        # handle expressions
        return units.get_units(var)
    else:
        # handle variables
        return var.get_units()


def get_attached_unit_str(var: Var | Expression | ExpressionBase) -> str:
    """
    Get the unit of a variable as a string.
    """
    unit = get_attached_unit(var)
    return str(unit) if unit is not None else "dimensionless"


def get_value(var: Var | Expression | ExpressionBase) -> float | dict:
    if var.is_indexed():
        if isinstance(var, Var):
            # can get values directly
            return cast(dict, var.get_values())
        else:
            data = {}
            for index in var.index_set():
                data[index] = value(var[index])
            return data
    else:
        return float(value(var))
