from .custom_valve import Valve as CustomValve, ValveFunctionType
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from .custom_pressure_changer import CustomPressureChanger

VALVE_FUNCTION_MAP = {
    "linear": ValveFunctionType.linear,
    "quick_opening": ValveFunctionType.quick_opening,
    "equal_percentage": ValveFunctionType.equal_percentage,
}

def ValveWrapper(**kwargs):
    enable_coefficients = kwargs.pop('enable_coefficients', False)
    valve_function = kwargs.pop('valve_function', None)

    # Set default valve function if none provided
    if valve_function is None:
        valve_function = "linear"

    # Check if valve_function is a string and map it to the corresponding callback
    if isinstance(valve_function, str):
        if valve_function in VALVE_FUNCTION_MAP:
            valve_function_callback = VALVE_FUNCTION_MAP[valve_function]
        else:
            raise ValueError(f"Unknown valve_function: {valve_function}")
    else:
        # just in case it is already a enum or a callback
        valve_function_callback = valve_function

    if enable_coefficients:
        # use the custom valve model in full
        return CustomValve(valve_function_callback=valve_function_callback, **kwargs)
    else:
        # add thermodynamic_assumption kwarg to
        # the PressureChanger model
        kwargs['thermodynamic_assumption'] = ThermodynamicAssumption.adiabatic
        kwargs['compressor'] = False
        # just use a pressure changer
        return CustomPressureChanger(**kwargs)
