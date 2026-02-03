from idaes.models.unit_models import Compressor, Heater
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback

from ahuora_builder_types import FlowsheetSchema
from ...generate_python_file import generate_python_code, PythonFileGenerator

GENERATE_TESTS = False

def test_generate_is_false():
    assert GENERATE_TESTS is False, "test_generate_python_file.py:  GENERATE_TESTS should be False to avoid overriding test_solved.json files during tests."


def test_import():
    generator = PythonFileGenerator({})
    heater_name, heater_import = generator.resolve_import(Heater)
    assert heater_name == "Heater"
    assert heater_import == "from idaes.models.unit_models.heater import Heater"
    compressor_name, compressor_import = generator.resolve_import(Compressor)
    assert compressor_name == "Compressor"
    assert compressor_import == "from idaes.models.unit_models.pressure_changer import Compressor"
    delta_t_name, delta_t_import = generator.resolve_import(delta_temperature_underwood_callback)
    assert delta_t_name == "delta_temperature_underwood_callback"
    assert delta_t_import == "from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback"


def test_names():
    # names need to be converted to valid python variable names
    generator = PythonFileGenerator({})
    def check_name(name: str, expected: str):
        # starts with m.fs.
        assert generator.get_name(name) == f"m.fs.{expected}"
    check_name("Pump-1", "Pump_1")
    check_name("Heater-1", "Heater_1")
    check_name("heat_exchanger-1", "heat_exchanger_1")
    check_name("name with spaces", "name_with_spaces")
    check_name("name-with-dashes", "name_with_dashes")
    check_name("name_with_underscores", "name_with_underscores")
    check_name("name_with_123_numbers", "name_with_123_numbers")
    check_name("123_numbers_at_start", "_123_numbers_at_start")
    check_name("name_with_!@#$%^&*()_special_characters", "name_with__special_characters")
    check_name("", "_unnamed_unit")
    check_name(" ", "_unnamed_unit")
    check_name("1", "_1")
    check_name(",", "_unnamed_unit")
    

def run_generate(input_file: str, expected_file: str) -> None:
    import os
    import json

    with open(os.path.join(os.path.dirname(__file__), input_file), "r") as f:
        flowsheet = json.load(f)
        flowsheet = FlowsheetSchema.model_validate(flowsheet)
    
    result = generate_python_code(flowsheet)

    # # write result to a file for debugging - or to regenerate expected files
    if GENERATE_TESTS:
        with open(os.path.join(os.path.dirname(__file__), expected_file), "w") as f:
            f.write(result)


    # split result line by line to compare each line
    lines = result.split("\n")

    with open(os.path.join(os.path.dirname(__file__), expected_file), "r") as f:
        expected_lines = f.readlines()
    
    # compare each line of the result with the expected file
    i = 0
    while len(lines) > 0 and len(expected_lines) > 0:
        # skip empty lines
        while len(lines) > 0 and lines[0].strip() == "":
            lines.pop(0)
        while len(expected_lines) > 0 and expected_lines[0].strip() == "":
            i += 1
            expected_lines.pop(0)
        if len(lines) == 0:
            assert len(expected_lines) == 0, f"Expected '{expected_lines[0]}'"
            break
        if len(expected_lines) == 0:
            assert len(lines) == 0, f"Got '{lines[0]}'"
            break
        # compare lines
        i += 1
        line = lines.pop(0).strip()
        expected_line = expected_lines.pop(0).strip()
        assert line == expected_line, f"Line {str(i)}: Expected '{expected_line}', got '{line}'"


def test_compressor():
    # basic test
    run_generate("../test_solver/configurations/compressor.json", "configurations/compressor_generated.py")


def test_heat_exchanger():
    # tests a few more features (constants, dictionaries)
    run_generate("../test_solver/configurations/heat_exchanger.json", "configurations/heat_exchanger_generated.py")


def test_pump_and_heater():
    # tests connected components
    run_generate("../test_solver/configurations/pump.json", "configurations/pump_generated.py")


def test_recycle():
    # tests tears
    run_generate("../test_solver/configurations/recycle.json", "configurations/recycle_generated.py")