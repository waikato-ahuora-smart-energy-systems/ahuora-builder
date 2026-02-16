from __future__ import annotations
from pyomo.core.base.var import ScalarVar
from pyomo.environ import units as pyunits
from ahuora_builder.methods.adapter import fix_slice, fix_var
from ahuora_builder.properties_manager import PropertyComponent

from pyomo.core.base.expression import ScalarExpression, Expression
from pyomo.core.base.var import ScalarVar, VarData,  _GeneralVarData
from pyomo.core.base.constraint import ScalarConstraint, ConstraintData, _GeneralConstraintData
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.environ import value
from sympy import var
from ahuora_builder.methods.adapter import fix_slice
from .units_handler import ValueWithUnits




def update_property(property_component: PropertyComponent, values: list[ValueWithUnits]):
    """
    This function is used to update what a property component is fixed to.
    For example, in the online solving, we want to update properties to their new values from the OPC UA server, and then re-solve the model.
    This saves us from having to re-create the model and re-fix all the properties every time.
    Args:
    - property_component (PropertyComponent): The property component to update. This can be retrieved from the PropertiesManager.
    - values (list[ValueWithUnits]): The new values to set the property component to. These could be a list of values with the IDAES unit attached too.

    """
    constraints: list[ScalarVar | ScalarConstraint] = list(property_component.corresponding_constraint)
    if len(values) != len(constraints):
        raise ValueError(f"Number of values provided ({len(values)}) does not match the number of constraints ({len(constraints)}) for property {property_component.name} (id {property_component}). Please provide the correct number of values.")
    
    free_property(property_component)

    new_constraints = fix_slice(property_component.component, values)
    property_component.corresponding_constraint = new_constraints



def free_property(property_component: PropertyComponent):
    components: list[ScalarVar | ScalarExpression] = list(property_component.component.values())
    constraints: list[ScalarVar | ScalarConstraint] = list(property_component.corresponding_constraint)
   
    if len(components) != len(constraints):
        raise ValueError(f"Number of components ({len(components)}) does not match the number of corresponding constraints ({len(constraints)}) for property {property_component.name} (id {property_component}). This is likely an issue with the model setup internal to ahuora-builder, please check how the property and its corresponding constraint are added to the model.")



    for component, constraint in zip(components, constraints):
        if not (isinstance(component, ScalarVar) or isinstance(component, ScalarExpression)):
            # If this gets raised we have a problem and need to look at the get_variable_from_property function
            raise ValueError(f"Something is wrong, we should have either a scalar variable or expression for this property_component, got {type(component)} for {component.name}")

        if isinstance(component, ScalarVar):
            if not component.is_fixed():
                raise ValueError(f"Variable {component.name} must be fixed before applying input value. It looks like this variable is not fixed in the flowsheet scenario. Please choose variables that are fixed, or update the scenario JSON")
            component.unfix()
        else: # ScalarExpression
            if constraint is None:
                raise ValueError(f"Could not find corresponding constraint for expression property {property_component.name}. This means that the expression is not an input. please choose properties that are fixed/inputs, or update the scenario JSON so that {component.name} is an input.")
            # Remove the constraint(s), so that we can add a new one for the OPC input value
            try:
                blk = constraint.parent_block()
                print(f"[DEL] Removing constraint: {constraint.name}")
                blk.del_component(constraint)
            except Exception as e:
                print(f"[ERR] Could not delete constraint {constraint} (for {property_component.name}): {e}")
                raise e
            
    