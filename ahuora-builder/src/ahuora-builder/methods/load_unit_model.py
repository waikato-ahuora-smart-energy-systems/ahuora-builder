from typing import Any

from pyomo.network import Port
from pyomo.environ import value as pyo_value, Var
from pyomo.core.base.constraint import ScalarConstraint
from pyomo.core.base.units_container import units
from idaes.core import UnitModelBlock
from idaes.core.util.tables import _get_state_from_port
from idaes.core.util.model_serializer import StoreSpec, from_json

from ahuora_builder.methods.BlockContext import BlockContext
from .adapter_library import UnitModelConstructor
from ..flowsheet_manager_type import FlowsheetManager
from ahuora_builder_types import  UnitModelSchema
from .adapter import fix_block
from ahuora_builder.methods.adapter import get_component

def add_unit_model_to_flowsheet(
    unit_model_def: UnitModelSchema,
    adapter_constructor: UnitModelConstructor,
    flowsheet_manager: FlowsheetManager,
) -> UnitModelBlock: 
    """
    Add the unit model to the flowsheet.
    Args:
        unit_model_def: The definition of the unit model to be added to the flowsheet.
        adapter_schema: Methods used to create the model and parse the arguments.
        flowsheet_manager: Flowsheet manager to interact with the flowsheet.
    """
    # Create the model
    idaes_model: UnitModelBlock =_create_model(unit_model_def, adapter_constructor, flowsheet_manager)

    # Add the model to the flowsheet
    component_name = f"{unit_model_def.name}_{unit_model_def.id}"
    flowsheet_manager.model.fs.add_component(component_name, idaes_model)

    # Import initial guesses
    initial_values = unit_model_def.initial_values

    if initial_values:
        from_json(idaes_model, initial_values, wts=StoreSpec.value())
    
    # Fix properties
    block_context = BlockContext(flowsheet_manager.model.fs)
    _fix_properties(idaes_model,unit_model_def, flowsheet_manager,block_context)
    # Fix ports
    _fix_ports(idaes_model, unit_model_def, flowsheet_manager,block_context)
    # Apply degrees of freedom (add constraints for controlled vars)
    block_context.apply_elimination()
    return idaes_model

def _create_model(unit_model_def : UnitModelSchema,adapter_constructor: UnitModelConstructor,flowsheet_manager : FlowsheetManager) -> UnitModelBlock:
    """
    Create the kwargs for the model constructor from the unit model definition.
    """
    arg_parsers = adapter_constructor.arg_parsers
    args = unit_model_def.args

    kwargs : dict[str, Any] = {}
    for name in args:
        if not name in arg_parsers:
            raise ValueError(
                f"Argument {name} not found in model schema, available arguments are {[x for x in arg_parsers.keys()]}"
            )
    for name in arg_parsers:
        register = arg_parsers[name]
        # try to get the argument from the schema passed in
        # however, in some cases the argument may not be required
        # i.e when specified by the model adapter itself
        # e.g methods.constant

        # note that in the future we may have to support passing
        # optional arguments to the model constructor
        result = register.run(args.get(name, None),flowsheet_manager)
        kwargs[name] = result

    idaes_model = adapter_constructor.model_constructor(**kwargs)
    return idaes_model

def _fix_properties( unit_model: UnitModelBlock, unit_model_def: UnitModelSchema, flowsheet_manager : FlowsheetManager,block_context: BlockContext) -> None:
        """
        Fix the properties of the unit model based on the properties in the unit model definition.
        """
        # Loop through the properties in the unit model definition to fix the properties in the unit model
        properties = unit_model_def.properties
        fix_block(
            unit_model, properties, flowsheet_manager.model.fs, block_context
        )


def _fix_ports(unit_model: UnitModelBlock, unit_model_def: UnitModelSchema, flowsheet_manager:FlowsheetManager,block_context: BlockContext) -> None:
        """
        Fix the ports of the unit model based on the ports in the unit model definition.
        """
        # Loop through the ports in the unit model definition to fix the ports in the unit model
        for port_name, port_schema in unit_model_def.ports.items():
            # Get the port from the unit model
            port = get_component(unit_model, port_name)
            if not isinstance(port, Port):
                raise ValueError(f"Port {port_name} not found in model")
            # Register the port, so arcs can connect to it by id
            flowsheet_manager.ports.register_port(port_schema.id, port)
            # Set the port parameters
            sb = _get_state_from_port(port, 0)
            state_block = sb.parent_component()

            # Prefer time-only properties_in/properties_out if available
            # decide by port name containing "inlet"/"outlet" and "hot"/"cold"
            pn = port_name.lower()
            if "inlet" in pn:
                if "hot" in pn and hasattr(unit_model, "hot_side") and hasattr(unit_model.hot_side, "properties_in"):
                    state_block = unit_model.hot_side.properties_in
                elif "cold" in pn and hasattr(unit_model, "cold_side") and hasattr(unit_model.cold_side, "properties_in"):
                    state_block = unit_model.cold_side.properties_in
            elif "outlet" in pn:
                if "hot" in pn and hasattr(unit_model, "hot_side") and hasattr(unit_model.hot_side, "properties_out"):
                    state_block = unit_model.hot_side.properties_out
                elif "cold" in pn and hasattr(unit_model, "cold_side") and hasattr(unit_model.cold_side, "properties_out"):
                    state_block = unit_model.cold_side.properties_out

            if sb.config.defined_state:
                """
                ie. Inlet state.

                The inlet state needs a separate context, because its variables
                are all fixed during initialization, so applying elimination
                involving variables outside the inlet state would run into
                degrees of freedom issues.
                """
                inlet_ctx = BlockContext(flowsheet_manager.model.fs)
                fix_block(
                    state_block,
                    port_schema.properties,
                    flowsheet_manager.model.fs,
                    inlet_ctx,
                )
                inlet_ctx.apply_elimination()
            else:
                # The outlet state(s) can use the unit model context.
                fix_block(
                    state_block,
                    port_schema.properties,
                    flowsheet_manager.model.fs,
                    block_context,
                )

    