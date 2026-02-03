from typing import Any
from pyomo.environ import Expression
from pyomo.core.base.units_container import units as pyomo_units, _PyomoUnit
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
from pyomo.core.expr.sympy_tools import sympy2pyomo_expression, PyomoSympyBimap
from pyomo.core.base.indexed_component import IndexedComponent
from ahuora_builder.properties_manager import PropertiesManager
from .slice_manipulation import is_scalar_reference
class ExpressionParsingError(Exception):
    """Custom exception for errors during expression parsing."""
    pass

# add extra units to pyomo's units library
# could probably make this on-demand
ureg = pyomo_units._pint_registry
ureg.define("dollar = [currency]")


def handle_special_chars(expr: str) -> str:
    # replace special characters so they can be parsed
    expr = expr.replace("^", "**")
    expr = expr.replace("$", "dollar")

    return expr


def get_property_from_id(fs, property_id,time_index):
    # returns a pyomo var from a property id
    properties_map : PropertiesManager = fs.properties_map
    pyomo_object: IndexedComponent = properties_map.get_component(property_id)
    
    if pyomo_object is None:
        raise ValueError(f"Symbol with id {id} not found in model")
    # check if this is a time-indexed var, and if so get the value at the given time index
    if is_scalar_reference(pyomo_object):
        # reference with index None
        return pyomo_object[None]
    elif pyomo_object.index_set() == fs.time:
        return pyomo_object[time_index]
    else:
        raise NotImplementedError("Only 0D and 1D time-indexed properties are supported in expressions")


def evaluate_symbol(fs, symbol: str,time_index) -> Any:
    if symbol.lower() == "time" or symbol.lower() == "t":
        return float(time_index)
    if symbol.startswith("id_"):
        # get the property from flowsheet properties_map
        id = int(symbol[3:])
        return get_property_from_id(fs, id,time_index)
    else:
        # assume its a unit, eg. "m" or "kg"
        # get the unit from pint, pyomo's units library
        ureg = pyomo_units._pint_registry
        pint_unit = getattr(ureg, symbol)
        pyomo_unit = _PyomoUnit(pint_unit, ureg)
        # We want people to write expressions such as (10 * W + 5 * kW). Pyomo doesn't natively support this,
        # so we can always convert to base units.
        if symbol == "delta_degC" or symbol == "delta_degF":
            # special case, because degC is not a base unit
            return 1 * pyomo_unit
            #return _PyomoUnit(ureg.delta_degC)
        elif symbol == "degC" or symbol == "degF":
            # throw an error (we do not support this, as it is unclear what to do)
            # https://pyomo.readthedocs.io/en/6.8.1/explanation/modeling/units.html
            raise ValueError(f"Use relative temperature units (delta_degC, delta_degF) or absolute temperature units (K, degF). Cannot use {symbol} as addition and multiplication is inconsistent on non-absolute units")
        scale_factor, base_units = ureg.get_base_units(pint_unit, check_nonmult=True) # TODO: handle degC etc.
        base_pyomo_unit = _PyomoUnit(base_units, ureg)
        return pyomo_units.convert( 1 * pyomo_unit, to_units=base_pyomo_unit)


class PyomoSympyMap(PyomoSympyBimap):

    def __init__(self, model,time_index):
        self.model = model
        self.time_index = time_index

    def getPyomoSymbol(self, sympy_object: Symbol, default=None):
        if not isinstance(sympy_object, Symbol):
            return None  # It's not in pyomo, e.g a number or something
        return evaluate_symbol(self.model, sympy_object.name, self.time_index)

    def getSympySymbol(self, pyomo_object, default=None):
        raise NotImplementedError(
            "getSympySymbol not implemented, because it shouldn't be needed"
        )
        # we don't care, it only needs to go one way

    def sympyVars(self):
        raise NotImplementedError(
            "sympyVars not implemented, because it shouldn't be needed"
        )


def parse_expression(expression, model,time_index) -> Expression:
    # use the bimap to get the correct pyomo object for each symbol
    bimap = PyomoSympyMap(model,time_index)
    try:
        expression = handle_special_chars(expression)
        sympy_expr = parse_expr(expression)
        pyomo_expr = sympy2pyomo_expression(sympy_expr, bimap)
    except Exception as e:
        raise ExpressionParsingError(f"{expression}: error: {e}")
    return pyomo_expr
