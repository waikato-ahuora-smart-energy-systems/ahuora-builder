from typing import Any
from idaes.core import FlowsheetBlock

from ahuora_builder_types.flowsheet_schema import PropertyPackageType
from ahuora_builder_types.payloads import BuildStateRequestSchema
from ahuora_builder_types.unit_model_schema import SolvedPropertyValueSchema
from .property_package_manager import create_property_package
from .methods.adapter import fix_block, serialize_properties_map, deactivate_fixed_guesses
from .methods.BlockContext import BlockContext
from pyomo.environ import ConcreteModel
from .properties_manager import PropertiesManager
from pyomo.environ import Block, assert_optimal_termination, SolverFactory
from pyomo.core.base.constraint import ScalarConstraint
from .flowsheet_manager import build_flowsheet
from idaes.core.util.model_statistics import degrees_of_freedom, number_unused_variables, number_activated_equalities, number_unfixed_variables, number_unfixed_variables_in_activated_equalities
from idaes.core.util.model_diagnostics import DiagnosticsToolbox

def solve_state_block(schema: BuildStateRequestSchema) -> list[SolvedPropertyValueSchema]:
    m, sb = build_state(schema.property_package)
    # if there's only one compound in the property package, remove the mole_frac_comp (as it's over specified)
    if len(schema.property_package.compounds) == 1:
        del schema.properties["mole_frac_comp"]
    
    block_ctx = BlockContext(m.fs)
    fix_block(sb, schema.properties, m.fs, block_ctx)

    # If the degrees of freedom are not zero, don't try solve.
    # However, the degree of freedom logic ignores any variables that aren't actually used. So if temperature 
    # and pressure are both not specified, and there are no constraints for them either, it decides that
    # they are not part of the solution, and says there's 0 degrees of freedom.
    # so instead, we actually check there are unfixed variables that are not in activated equalities.
    # TODO: Check why adding and degrees_of_freedom(sb) == 0 makes pr fail (python manage.py test core.auxiliary.tests.test_Compounds)
    if number_unfixed_variables(sb) - number_unfixed_variables_in_activated_equalities(sb) != 0:
        return [] # This means no properties are returned, so the backend won't update anything.
    block_ctx.apply_elimination()
    
    # initialise the state block, which will perform a solve
    sb.initialize(outlvl=1)
    deactivate_fixed_guesses(m.fs.guess_vars)

    return serialize_properties_map(m.fs)


def build_state(schema: PropertyPackageType) -> Any:  # PropertyPackageSchema
    m = build_flowsheet(dynamic=False)
    # create the property package and state block
    property_package = create_property_package(schema, m)
    state_block = property_package.build_state_block(m.fs.time, defined_state=True)
    m.fs.add_component(f"PP_{schema.id}_state", state_block)

    return m, state_block


def get_state_vars(schema: PropertyPackageType) -> Any:
    _, state_block = build_state(schema)

    return state_block[0].define_state_vars()
