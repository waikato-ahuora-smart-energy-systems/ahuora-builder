from typing import Any
from enum import Enum
from pyomo.core.base.units_container import units
from pyomo.environ import value as get_value
from ahuora_builder_types import FlowsheetSchema, UnitModelSchema, PropertiesSchema, PortsSchema
from ahuora_builder_types.flowsheet_schema import PropertyPackageType
from .methods.units_handler import idaes_specific_convert, attach_unit, get_attached_unit, get_attached_unit_str
from .methods.adapter_library import AdapterLibrary, UnitModelConstructor
from .methods import adapter_methods
from .build_state import get_state_vars
import json
from ahuora_builder.custom.energy.power_property_package import PowerParameterBlock



class Section:
    """
    Represents a section of a Python file
    Includes a name and a list of lines
    """
    def __init__(self, name: str, header: bool = True, new_line: bool = True, optional: bool = False) -> None:
        self._name: str = name
        self._lines: list[str] = []
        self._header: bool = header
        self._new_line: bool = new_line
        self._optional: bool = optional
    
    def extend(self, lines: list[str]) -> None:
        self._lines.extend(lines)
    
    def header(self) -> str:
        return f"### {self._name}"
    
    def lines(self) -> list[str]:
        return self._lines


class PythonFileGenerator:
    """
    Generate a Python file from the given model data
    """

    def __init__(self, schema: FlowsheetSchema) -> None:
        self._schema = schema
        
        # set some global constants
        self._model = "m"
        self._flowsheet = "fs"
        self._solver = "ipopt"
        # store the unit models, property packages, and ports
        self._property_packages: dict[int, dict[str, Any]] = {}  # {id: {name, vars: {vars}}}
        self._ports: dict[int, dict[str, Any]] = {}  # {id: {name, arc_id}}
        self._arcs: dict[int, dict[str, Any]] = {}  # {id: {name, source_id, destination_id}}

        # create the sections
        self._sections = {
            "imports": Section("Imports"),
            "property_package_imports": Section("Property Package Imports", header=False, new_line=False),
            "unit_model_imports": Section("Unit Model Imports", header=False, new_line=False),
            "utility methods": Section("Utility Methods"),
            "build_model": Section("Build Model"),
            "create_property_packages": Section("Create Property Packages", header=False),
            "create_unit_models": Section("Create Unit Models", header=False),
            "create_arcs": Section("Connect Unit Models", optional=True),
            "check_model": Section("Check Model Status"),
            "initialize": Section("Initialize Model"),
            "solve": Section("Solve"),
            "report": Section("Report"),
        }


    def sections(self) -> dict:
        return self._sections
    

    def setup_sections(self) -> None:
        """
        Set up the sections with the initial (constant) lines
        """
        self.extend("imports", [
            "from pyomo.environ import ConcreteModel, SolverFactory, SolverStatus, TerminationCondition, Block, TransformationFactory, assert_optimal_termination",
            "from pyomo.network import SequentialDecomposition, Port, Arc",
            "from pyomo.core.base.units_container import _PyomoUnit, units as pyomo_units",
            "from idaes.core import FlowsheetBlock",
            "from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom",
            "from idaes.core.util.tables import _get_state_from_port",
            "import idaes.logger as idaeslog",
        ])
        self.extend("property_package_imports", [
            "from property_packages.build_package import build_package",
        ])
        self.extend("utility methods", [
            "def units(item: str) -> _PyomoUnit:",
            "    ureg = pyomo_units._pint_registry",
            "    pint_unit = getattr(ureg, item)",
            "    return _PyomoUnit(pint_unit, ureg)",
        ])
        self.extend("build_model", [
            f"{self._model} = ConcreteModel()",
            f"{self._model}.{self._flowsheet} = FlowsheetBlock(dynamic=False)",
        ])
        self.extend("create_property_packages", [
            "# Set up property packages",
        ])
        self.extend("create_unit_models", [
            "# Create unit models",
        ])
        self.extend("check_model", [
            f"report_statistics({self._model})",
            f"print(\"Degrees of freedom:\", degrees_of_freedom({self._model}))",
        ])
        self.extend("solve", [
            f"opt = SolverFactory(\"{self._solver}\")",
            f"res = opt.solve({self._model}, tee=True)",
            f"assert_optimal_termination(res)",
        ])
    

    def extend(self, section_name: str, lines: list[str] | str) -> None:
        # add lines to a section
        if isinstance(lines, str):
            lines = [lines]
        self._sections[section_name].extend(lines)
    

    def add_section_excl(self, section_name: str, line: str) -> None:
        # add a line to a section if it is not already present
        if line not in self._sections[section_name].lines():
            self.extend(section_name, line)


    def resolve_import(self, obj: type | Enum) -> tuple[str, str]:
        # get the class name and import statement for a given class
        module_name = obj.__module__
        if isinstance(obj, Enum):
            import_name = obj.__class__.__name__  # name to be used in the import statement
            class_name = f"{obj.__class__.__name__}.{obj.name}"  # name to be used in the code
        else:
            import_name = obj.__name__
            class_name = import_name
        return class_name, f"from {module_name} import {import_name}"
    

    def get_name(self, name: str) -> str:
        # clean the name so that it can be used as a Python variable
        name = name.strip().replace("-", "_").replace(" ", "_")
        # remove any spaces and special characters
        name = "".join([char for char in name if char.isalnum() or char == "_"])
        # if the name is empty, use a default name
        if len(name) == 0:
            name = f"_unnamed_unit"
        # if the name starts with a number, add an underscore
        if name[0].isnumeric():
            name = "_" + name
        name = f"{self._model}.{self._flowsheet}.{name}"
        return name
    

    def create_property_packages(self) -> None:
        """
        Create property packages
        """
        for schema in self._schema.property_packages:
            self.create_property_package(schema)


    def create_property_package(self, schema: PropertyPackageType) -> None:
        name = self.get_name("PP_" + str(schema.id))
        compounds = schema.compounds
        phases = schema.phases
        type = schema.type
        self.extend("create_property_packages", [
        f"{name} = build_package(",
        f"    \"{type}\",",
        f"    {json.dumps(compounds)},",
        ")",
        ])
        self._property_packages[schema.id] = { "name": name, "vars": get_state_vars(schema) }


    def get_property_package(self, id: int) -> dict:
        """Get the name of a property package by ID"""
        if id == -1 and id not in self._property_packages:
            # add a default Helmholtz property package (for testing purposes)
            self.create_property_package(PropertyPackageType(id=id, type="helmholtz", compounds=["h2o"], phases=["Liq"]))
        return self._property_packages[id]
    
    def get_power_property_package(self, id: str):
        power_package = PowerParameterBlock
        return self._property_packages[-1]


    def get_property_package_at_port(self, model_schema: UnitModelSchema, port: str) -> dict:
        """Get the property package that is used at a port"""
        # generally, the unitop has one property package that is used for all ports
        if model_schema.args.get("property_package") is not None:
            return self.get_property_package(model_schema.args["property_package"])
        # this is a special case (hard-coded for now) for heat exchangers, which have two property packages
        # TODO: make this dynamic once we have parent stream inheritance
        package_arg = port.removesuffix("_inlet").removesuffix("_outlet")  # eg. "hot_side_inlet" -> "hot_side"
        return self.get_property_package(model_schema.args[package_arg]["property_package"])
    

    def serialise_dict(self, d: dict, indent: bool = False, indent_level: int = 1, nested_indent: bool = True) -> str:
        if len(d) == 0:
            return "{}"
        result = "{"
        for k, v in d.items():
            if indent:
                result += "\n" + "    " * indent_level
            adj_k = f"\"{k}\"" if isinstance(k, str) else k
            adj_v = f"\"{v}\"" if isinstance(v, str) and not v.startswith(f"{self._model}.{self._flowsheet}.") else v
            if isinstance(v, dict):
                # allow nested dictionaries
                adj_v = self.serialise_dict(v, indent=indent and nested_indent, indent_level=indent_level + 1)
            result += f"{adj_k}: {adj_v},"
        result = result[:-1] + ("\n" + "    " * (indent_level - 1)) * indent + "}"
        return result
    

    def serialise_list(self, l: list) -> str:
        if len(l) == 0:
            return "[]"
        result = "["
        for v in l:
            adj_v = f"\"{v}\"" if isinstance(v, str) and not v.startswith(f"{self._model}.{self._flowsheet}.") else v
            result += f"{adj_v},"
        result = result[:-1] + "]"
        return result
    

    def setup_args(self, args: dict, arg_parsers: dict) -> dict:
        """Setup the arguments for a unit model"""
        result: dict[str, Any] = {}
        print("args: " + str(args))
        for arg_name, method in arg_parsers.items():
            def match_method() -> Any:
                match method.__class__:
                    case adapter_methods.Constant:
                        # constant, defined in the method
                        constant = method.run(None,None)
                        # constant can be a function, in which case we need to resolve the import
                        if callable(constant) or isinstance(constant, Enum):
                            constant_class_name, constant_import = self.resolve_import(constant)
                            self.add_section_excl("imports", constant_import)
                            return constant_class_name
                        return constant
                    case adapter_methods.Value:
                        # value, keep as is
                        return args.get(arg_name, None)
                    case adapter_methods.PropertyPackage:
                        # property package
                        property_package_id = args["property_package"]
                        print(property_package_id)
                        return self.get_property_package(property_package_id)["name"]
                    case adapter_methods.PowerPropertyPackage:
                        #power property package
                        return "m.fs.power_property_package"
                    
                    case adapter_methods.Dictionary:
                        # another dictionary of arg parsers, recursively setup the args
                        return self.setup_args(args[arg_name], method._schema)
                    case _:
                        raise Exception(f"Method {method} not supported")
            result[arg_name] = match_method()
        return result


    def write_args(self, args: dict) -> str:
        args_str = ""
        if len(args) == 0:
            return args_str
        args_str += "\n"
        for key, value in args.items():
            args_str += f"    {key}="
            if isinstance(value, dict):
                args_str += self.serialise_dict(value)
            else:
                args_str += f"{value}"
            args_str += ",\n"
        args_str = args_str[:-2] + "\n"  # remove the last comma
        return args_str
    

    def create_unit_models(self) -> None:
        """
        Create the unit models
        """
        for unit_model in self._schema.unit_models:
            # add to imports
            adapter: Adapter = AdapterLibrary[unit_model.type]
            class_name, class_import = self.resolve_import(adapter.model_constructor)
            self.add_section_excl("unit_model_imports", class_import)
            # setup args
            args = self.setup_args(unit_model.args, adapter.arg_parsers)
            args_str = self.write_args(args)
            print("args_str: " + args_str)

            # create the unit model
            name = self.get_name(unit_model.name)
            self.extend("create_unit_models", f"\n# {unit_model.name}")  # comment
            self.extend("create_unit_models", f"{name} = {class_name}({args_str})")  # constructor
            self.extend("create_unit_models", self.fix_properties(name, unit_model.properties))  # fix properties
            for port_name, port_data in unit_model.ports.items():
                # save the port
                global_name = f"{name}.{port_name}"
                self._ports[port_data.id] = { "name": global_name, "arc": None }
                # available_vars = self.get_property_package_at_port(unit_model, port_name)["vars"]
                # fix the properties of the port
                self.extend("create_unit_models", self.fix_state_block(global_name, port_data.properties))
            self.extend("report", f"{name}.report()")
    

    def fix_properties(self, prefix: str, properties_schema: PropertiesSchema) -> list[str]:
        lines = []
        for key, property_info in properties_schema.items():
            for property_value in property_info.data:
                if property_value.value is None:
                    continue
                if property_value.discrete_indexes is not None:
                    indexes_tuple = tuple(property_value.discrete_indexes)
                    indexes_string = f"[{indexes_tuple}]" if len(property_value.discrete_indexes) > 0 else ""
                else:
                    indexes_string = ""
                val = get_value(property_value.value)
                unit = property_info.unit
                # TODO: Handle dynamic indexes etc.
                lines.append(f"{prefix}.{key}{indexes_string}.fix({val} * units(\"{unit}\"))")
                    
        return lines

    
    def fix_state_block(self, prefix: str, properties: PropertiesSchema) -> list[str]:
        """
        Fix the properties of a unit model
        """
        lines = []
        lines.append(f"sb = _get_state_from_port({prefix}, 0)")
        for key, property_info in properties.items():
            for property_value in property_info.data:
                if property_value.value is None:
                    continue
                if property_value.discrete_indexes is None:
                    indexes_str = ""
                else:
                    # We aren't worrying about time yet, but we will need to do in the future.
                    indexes_tuple = tuple(property_value.discrete_indexes)
                    indexes_str = f"[{indexes_tuple}]" if len(property_value.discrete_indexes) > 0 else ""
                val = get_value(property_value.value)
                unit = property_info.unit
                lines.append(f"sb.constrain_component(sb.{key}{indexes_str}, {val} * units(\"{unit}\"))")
                    
        if len(lines) == 1:
            return []
        return lines
    

    def create_arcs(self) -> None:
        """
        Create the arcs
        """
        if len(self._schema.arcs) == 0:
            return
        for i, arc in enumerate(self._schema.arcs):
            source = self._ports[arc.source]
            destination = self._ports[arc.destination]
            name = f"{self._model}.{self._flowsheet}.arc_{i + 1}"
            self.extend("create_arcs", f"{name} = Arc(source={source['name']}, destination={destination['name']})")
            self._arcs[i] = { "name": name, "source": arc.source, "destination": arc.destination }
            source["arc"] = i
            destination["arc"] = i
        # add to initialization: expand the arcs
        self.extend("initialize", f"TransformationFactory(\"network.expand_arcs\").apply_to({self._model})")
    

    def initialize(self) -> None:
        """
        Initialize the model
        """
        def is_connected(ports: PortsSchema) -> bool:
            for _, port_data in ports.items():
                port = self._ports[port_data.id]
                if port["arc"] is not None:
                    return True
            return False
        # initialize everything that is not connected
        for unit_model in self._schema.unit_models:
            if not is_connected(unit_model.ports):
                name = self.get_name(unit_model.name)
                self.extend("initialize", f"{name}.initialize(outlvl=idaeslog.INFO)")
        if len(self._schema.arcs) == 0:
            return
        # setup sequential decomposition
        self.extend("utility methods", [
            "\ndef init_unit(unit: Block) -> None:",
            "    unit.initialize(outlvl=idaeslog.INFO)"
        ])
        self.extend("initialize", "seq = SequentialDecomposition()")
        # set tear guesses
        tear_set = []

        # Need to rewrite the logic for dealing with tear sets from recycle

        self.extend("initialize", f"seq.set_tear_set({self.serialise_list(tear_set)})")
        self.extend("initialize", f"seq.run({self._model}, init_unit)")
                

def generate_python_code(model_data: FlowsheetSchema) -> str:
    """
    Generate a Python file from the given model data
    """
    generator = PythonFileGenerator(model_data)
    generator.setup_sections()
    generator.create_property_packages()
    generator.create_unit_models()
    generator.create_arcs()
    generator.initialize()
    sections = generator.sections()

    result = ""
    for key, section in sections.items():
        if section._optional and len(section.lines()) == 0:
            # skip empty sections
            continue
        # add extra newline characters between sections
        if section._new_line and key != list(sections.keys())[0]:
            if section._header:
                result += "\n\n"
            else:
                result += "\n"
        # add section header
        if section._header:
            result += section.header()
            if section._new_line:
                result += "\n"
        # write each line to the result string
        # separated by a newline character
        result += "\n".join(section.lines()) + "\n"
    return result
    