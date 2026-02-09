import os
import json
import io

from typing import Any

from ahuora_builder_types import FlowsheetSchema
from ahuora_builder_types.payloads import BuildStateRequestSchema
from ahuora_builder.flowsheet_manager import FlowsheetManager
from ahuora_builder.build_state import solve_state_block
from contextlib import redirect_stdout


# Set this to True to regenerate expected output files, if needed (e.g output file format changes.)
# Make sure to review the commit to ensure that only intended changes are made to expected output files.
GENERATE_TESTS = False

# Get current location (so that we can retrieve pump.json)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def try_solve(input_file: str, expected_output_file: str, state_block=False) -> None:
    """Tests that the model can be solved and that the answer returned is correct"""

    with open(os.path.join(__location__, input_file), 'r') as file:

        data = json.load(file)

        if state_block:
            build_state_schema = BuildStateRequestSchema.model_validate(data)
            result = solve_state_block(build_state_schema)
            result = [solved_value.model_dump() for solved_value in result]
        else:
            flowsheet_schema = FlowsheetSchema.model_validate(data)
            flowsheet = FlowsheetManager(flowsheet_schema)
            flowsheet.load()
            flowsheet.initialise()
            assert flowsheet.degrees_of_freedom() == 0, "Degrees of freedom is not 0: " + str(flowsheet.degrees_of_freedom())
            flowsheet.report_statistics()
            flowsheet.solve()
            flowsheet.optimize()

            result = flowsheet.serialise()
            result = result.model_dump()

            # ignore initial values
            del result["initial_values"]

        # # write result to a file for debugging - or to regenerate expected files
        if GENERATE_TESTS:
            with open(os.path.join(__location__, expected_output_file), 'w') as f:
                f.write(json.dumps(result, indent=4))

        print(result)

        # Assert - Check if all the parameters are correct
        with open(os.path.join(__location__, expected_output_file), 'r') as file:
            # Load the result JSON file
            expected_result = json.load(file)

            # Check if the result is equal to the expected_result
            try:
                if state_block:
                 values_approximately_equal(result, expected_result, 1e-1)
                else:
                    values_approximately_equal(result, expected_result, 1e-1)
            except Exception as e:
                print(e)
                raise AssertionError(f"Did not solve the model correctly")


def values_approximately_equal(val1: Any, val2: Any, tolerance: float = 1e-3) -> bool:
    """
    Checks if two values are approximately equal
    """
    if isinstance(val1, (float, int)) and isinstance(val2, (float, int)):
        if not abs(val1 - val2) < tolerance:
            raise AssertionError(f"Value: {val1} is not equal to expected value: {val2} within tolerance {tolerance}")
    elif isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            raise AssertionError(f"Lists have different lengths: {len(val1)} and {len(val2)}")
        return all(values_approximately_equal(v1, v2, tolerance) for v1, v2 in zip(val1, val2))
    elif isinstance(val1, dict) and isinstance(val2, dict):
        if set(val1.keys()) != set(val2.keys()):
            raise AssertionError(f"Keys: {val1.keys()} are not equal to expected keys: {val2.keys()}")
        return all(values_approximately_equal(val1[key], val2[key], tolerance) for key in val1)
    else:
        if val1 != val2:
            raise AssertionError(f"Value: {val1} is not equal to expected value: {val2}")
    return True


def test_generate_is_false():
    assert GENERATE_TESTS is False, "test_generate_python_file.py:  GENERATE_TESTS should be False to avoid overriding test_solved.json files during tests."


def test_solve_pump_and_heater() -> None:
    # Tests that the pump and heater model can be solved 
    try_solve('configurations/pump.json', 'configurations/pump_solved.json')


def test_solve_recycle() -> None:
    # Tests that the recycle model can be solved (Recycle connected to itself)
    try_solve('configurations/recycle.json', 'configurations/recycle_solved.json')


def test_solve_compressor() -> None:
    # Tests that the compressor model can be solved
    try_solve('configurations/compressor.json', 'configurations/compressor_solved.json')


def test_solve_mixer() -> None:
    # Tests that the mixer model can be solved
    try_solve('configurations/mixer.json', 'configurations/mixer_solved.json')


def test_solve_bt_pr() -> None:
    # Peng-Robinson test with Benzene and Toluene
    try_solve('configurations/BT_PR.json', 'configurations/BT_PR_solved.json')


def test_solve_heat_exchanger() -> None:
    """Tests that the BT_PR model can be solved"""
    try_solve('configurations/heat_exchanger.json','configurations/heat_exchanger_solved.json')


def test_solve_heat_pump() -> None:
    """Tests that the turbine model can be solved"""
    # TODO: fix this test case - the values specified are not very good
    # and although the model solves, it doesn't really show anything useful
    try_solve('configurations/heat_pump.json','configurations/heat_pump_solved.json')


def test_solve_unit_conversions() -> None:
    """
    Checks that unit conversions work as expected

    To do this, we solve the pump model and check that everything works as expected
    """
    # the output file should stay the same since we are just changing the units
    try_solve('configurations/pump_unit_conversions.json', 'configurations/pump_solved.json')


def test_solve_expressions() -> None:
    """
    Checks that expressions work as expected
    """
    try_solve('configurations/expressions.json', 'configurations/expressions_solved.json')


def test_solve_vapor_frac_target():
    # compressor_test_with_vapor_fraction - targets an outlet vapor fraction of 1
    try_solve('configurations/vapor_frac_target.json', 'configurations/vapor_frac_target_solved.json')


def test_solve_mass_flow_tear():
    # recycle with mass flow constraint
    try_solve('configurations/mass_flow_tear.json', 'configurations/mass_flow_tear_solved.json')


def test_solve_propane_recycle():
    try_solve('configurations/propane_recycle.json', 'configurations/propane_recycle_solved.json')


def test_solve_propane_heat_pump():
    try_solve('configurations/propane_heat_pump.json', 'configurations/propane_heat_pump_solved.json')


def test_solve_control():
    # inlet enthalpy is controlled by the set point of the outlet enthalpy
    try_solve('configurations/control.json', 'configurations/control_solved.json')
    
def test_solve_machine_learning():
    try_solve('configurations/machine_learning.json', 'configurations/machine_learning_solved.json')


def test_solve_sb_vapor_frac() -> None:
    # Tests that a state block with vapor_frac can be solved
    try_solve('configurations/sb_vapor_frac.json', 'configurations/sb_vapor_frac_solved.json', state_block=True)


def test_solve_compound_separator() -> None:
    # Tests that a compound separator model can be solved
    try_solve('configurations/compound_separator.json', 'configurations/compound_separator_solved.json')


def test_solve_constraints() -> None:
    # Tests that a model with constraints can be solved
    # Tests for an equality constraint using an expression
    try_solve('configurations/constraints.json', 'configurations/constraints_solved.json')


def test_solve_link() -> None:
    # Tests that a bus model can be solved
    try_solve('configurations/bus.json', 'configurations/bus_solved.json')


def test_solve_optimization() -> None:
    # Tests optimizing an expression
    try_solve('configurations/optimization.json', 'configurations/optimization_solved.json')

def test_solve_solar() -> None:
    # Tests that a solar model can be solved  
    try_solve('configurations/solar.json', 'configurations/solar_solved.json') 


def test_solve_milk_heater() -> None:
    try_solve('configurations/milk_heater.json', 'configurations/milk_heater_solved.json') 

def test_solve_custom_property_package() -> None:
    try_solve('configurations/custom_property_package.json', 'configurations/custom_property_package_solved.json') 

def test_solve_dynamic_tank() -> None:
    try_solve('configurations/dynamic_tank.json', 'configurations/dynamic_tank_solved.json') 

def test_solve_elimination() -> None:
    # more complex elimination, where there are choices to eliminate and the right pair has to be eliminated
    # ie. because we can't have both enthalpy and temperature fixed at the same time, leaving
    # both flow_mol and flow_mass unfixed (even this is still the correct degrees of freedom)
    # in this specific test case, flow_mass should eliminate flow_mol and constrain at the state block level
    try_solve('configurations/elimination.json', 'configurations/elimination_solved.json')

def test_solve_header() -> None:
    # Tests that a compound separator model can be solved
    try_solve('configurations/header.json', 'configurations/header_solved.json')

def test_initialisation_disabled_branch():
    """
    Test that FlowsheetManager.initialise() skips initialisation and prints the correct message
    when disable_initialization is True.
    """
    schema = FlowsheetSchema(
        id=999,
        dynamic=False,
        time_set=[0],
        property_packages=[],
        unit_models=[],
        arcs=[],
        expressions=[],
        optimizations=[],
        is_rating_mode=False,
        disable_initialization=True,  # Disable initialisation for this test
    )
    manager = FlowsheetManager(schema)
    manager.load()

    f = io.StringIO()
    with redirect_stdout(f):
        manager.initialise()
    output = f.getvalue()

    assert "Initialisation is disabled for this scenario." in output
    assert "initialise_model" not in output