import traceback
from typing import Any

from pyomo.environ import Block, value as pyo_value, Component, Var, ScalarVar, Reference, Expression
from pyomo.core.base.expression import ScalarExpression
from pyomo.core.base.constraint import ScalarConstraint
from pyomo.core.base.units_container import units
from idaes.core import FlowsheetBlock
from ahuora_builder.properties_manager import PropertiesManager
from ahuora_builder_types.unit_model_schema import SolvedPropertyValueSchema
from .units_handler import attach_unit, get_attached_unit, get_attached_unit_str, ValueWithUnits
from ahuora_builder_types import PropertiesSchema, PropertySchema
from typing import Dict
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import (
    IndexedComponent_slice,
)
import numpy as np
from ahuora_builder_types.id_types import PropertyValueId

def get_component(blk: Block, key: str):
    """
    Get a component from a block, given a key. Doesn't handle indexes.
    """
    try:
        # allow key to be split by "." to access nested properties eg. "hot_side.deltaP"
        key_split = key.split(".")
        b = blk
        for k in key_split:
            b = getattr(b, k)
        return b
    except AttributeError:
        raise ValueError(
            f"Property {key} not found in block `{blk}`. "
            f"Available properties are {[x for x in blk.component_map().keys()]}"
        )

def add_to_property_map(vars: IndexedComponent, id: PropertyValueId, fs):
    try:
        name = next(vars.values()).name # maybe we should deprectate the name?
        fs.properties_map.add(id, vars, name)
    except StopIteration:
        pass # The only reason this would happen is if there are no values in the indexed component, e.g in milk solids, there is no gas phase. In these case, we just skip adding to the property map. 
    #return c

def add_corresponding_constraint(fs: FlowsheetBlock,c, id: PropertyValueId):
    fs.properties_map.add_constraint(id, c)


def soft_cast_float(value: Any) -> float:
    """
    Softly cast a value to float, returning None if the value is None or cannot be cast.
    This is needed because not everything is indexed by time at all.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def items_by_time(s: Dict[str, Any]) -> list[tuple[str, Any]]:
    """
    Converts a dictionary of items indexed by time (as strings) to a list of tuples, and fixes the ordering.
    This is because if "11" is ordered before "2" in a dictionary of strings, but we want the results ordered by time.
    """
    return sorted(s.items(), key=lambda x: soft_cast_float(x[0]))  # sort by time index



def get_index_set_shape(component: IndexedComponent) -> tuple[int,...]:
    index_set = component.index_set()
    if index_set.dimen == 0:
        return () # I don't think we ever actually have this case (we early return if there are no items in the index set with Warning: No variables found)
    elif index_set.dimen == 1:
        return (len(index_set),)
    elif index_set.dimen > 1:
        shape = [len(s) for s in index_set.subsets()]
        return tuple(shape)


def serialize_properties_map(fs: FlowsheetBlock) -> list[SolvedPropertyValueSchema]:
    properties : list[SolvedPropertyValueSchema] = []
    # TODO: collate properties by timestep.
    properties_map: PropertiesManager = fs.properties_map
    for (uid, s) in properties_map.items():
        if uid == -1:
            # skip
            continue
        s.component

        shape = get_index_set_shape(s.component)

        values = [pyo_value(c) for c in s.component.values()]

        if shape == ():
            # not indexed at all, just return the value (not as an array)
            items = values[0]
        else:
            # Convert to an ndarray
            items = np.array(values).reshape(shape).tolist()
 
        property_dict = SolvedPropertyValueSchema(
            id=uid,
            name=s.name,  # for debugging
            value=items,
            unit=get_attached_unit_str(s.component),
        )
        if s.unknown_units:
            property_dict.unknown_units = s.unknown_units

        properties.append(property_dict)

    return properties

def slice_is_indexed(blk: IndexedComponent | IndexedComponent_slice) -> bool:
    """
    Check if a block, variable, or expression is not indexed.
    """
    if isinstance(blk, IndexedComponent_slice):
        # IndexedComponent_slice doesn't have is_indexed() method.
        # Instead, it runs is_indexed() on each of the underlying components.
        # We can assume that they are all the same, and just check the first one.
        return blk.is_indexed()[0]
    return blk.is_indexed()

def slice_index_dimen(blk: IndexedComponent | IndexedComponent_slice) -> int:
    """
    Get the dimension of the index set of a block, variable, or expression.
    """
    if isinstance(blk, IndexedComponent_slice):
        # IndexedComponent_slice doesn't have index_set() method.
        # Instead, it runs index_set() on each of the underlying components.
        # We can assume that they are all the same, and just check the first one.
        return blk.index_set()[0].dimen
    return blk.index_set().dimen


def get_sliced_version(block: Block | Var | Expression | IndexedComponent_slice) -> Block | Var | Expression | IndexedComponent_slice:
    """
    Get a sliced version of a block, variable, or expression.
    A very loose way of thinking of a sliced version is it "references all the indexes at the same time".
    
    in a scalar block, you can do scalar_block.property.
    in an indexed block, you can't do indexed_block.property, because you need to define the index you're looking at.
    However, if you want to define the indexes later, such as in a reference, you can do indexed_block[:].property.
    I.e you're creating a slice to all the indexes, and then accessing the property on that slice.

    You can't really use a sliced version directly, but you can use it to create a Reference with reference(sliced_version).
    It'll put all the indexes back together.

    The main advantage is that you can use get_sliced_version on a indexed subattribute of a slice, and it will collate the indexes together.
    """
    if not slice_is_indexed(block):
        return block
    
    block_dimen = slice_index_dimen(block)
    # we want to get a slice to all items in a block, like block[:,:]
    # To do this programmatically:
    # the ":" is represented by slice(None)
    # so for a 2D block, we want (slice(None), slice(None))
    # we can use a tuple comprehension to create this for any dimen
    block_slice = block[tuple(slice(None) for _ in range(block_dimen))]
    return block_slice


def collate_indexes(block: Block, property_key: str) -> IndexedComponent:
    """
    Returns a reference to the property with all the indexes together.
    This is because some properties have a different way of doing indexes:
    e.g 
     - properties_out[t].temperature   -> Reference[t]
     - block.heat_duty[t] -> Reference[t]
     - properties_out[t].mole_frac_comp[compound] -> Reference[t, compound]
     - block.area -> Reference[None]
    """
    block_slice = get_sliced_version(block)
    # Now we have the sliced version of the block, we can access the property using the property key.
    # As we support nested properties, e.g property_key could be "hot_side.deltaP",
    # we can use the get_component function rather than just getattr.
    block_property = get_component(block_slice, property_key)
    if block_property is None:
        raise ValueError(f"Property {property_key} not found in block {block}.")
    # Now we have the property, we need to check if it has any indexes, and get a slice to all those indexes.
    property_slice = get_sliced_version(block_property)
    #property_slice = block_property
    # Calling is_indexed() on a property_slice doesn't really make sense, it returns an array of bools. slices are always indexed though so
    # that array gets cast to True, so its' kinda okay.
    if property_slice.is_indexed(): # and not isinstance(property_slice, IndexedComponent_slice):
        # if the property is indexed, we need to return a reference to the slice.
        # This will collate all the indexes together.
        # Note that we need to special case DerivativeVar, as Reference doesn't support DerivativeVar directly. ( pyomo dae will try differentiate it again.)
        return Reference(property_slice, ctype=IndexedComponent) ##  if isinstance(block_property,DerivativeVar)  else NOTSET
    else: # this is not indexed at all lol, so we can just return the property itself. Creating a reference will add a index[None] which is unnecessary.
        return property_slice
        


def fix_var(blk: Block, var : ScalarVar | ScalarExpression, value : ValueWithUnits) -> ScalarVar | ScalarConstraint:#
    # returns: the expression that a constraint was added for, or the fixed var
    if hasattr(blk, "constrain_component"):
        component = blk.constrain_component(var, value)
    elif isinstance(var, ScalarExpression):
        component = var
    else:
        #print(var)
        var.fix(value)
        component = var
    return component

def fix_slice(var_slice: IndexedComponent | IndexedComponent_slice, values: list[ValueWithUnits]) -> list[ScalarVar | ScalarConstraint]:
    # Fix a slice of a variable to the given values.
    # the var_slice should be a Reference or other indexed component, or a scalar variable
    # values should be a list of values to fix the variable to.
    results: list[ScalarVar | ScalarConstraint] = []
    for var, value in zip(var_slice.values(), values):
        blk = var.parent_block()
        constraint = fix_var(blk, var, value)
        results.append(constraint)
    return results


def deactivate_components(components: list[ScalarVar | ScalarConstraint ]):
    # Deactivate a "PropertyValue" (which may have multiple subcomponents if it's indexed)
    for c in components:
        deactivate_component(c)

def deactivate_component(c: ScalarVar | ScalarConstraint):
    # deactivate "guess" variables: fixed for initialisation
    # and unfixed for a control constraint
    if isinstance(c, ScalarConstraint):
        c.deactivate()
    else:
        c.unfix()

def deactivate_fixed_guesses(guess_vars: list[list[ScalarVar | ScalarConstraint]]):

    for c in guess_vars:
        deactivate_components(c)


def load_initial_guess(component: Component, value: float):
    # load the initial guess into the component
    if isinstance(component, Var):
        component.value = value

def load_initial_guesses(components: IndexedComponent, values: list[float]):
    for c, v in zip(components.values(), values):
        load_initial_guess(c, v)


def fix_block(
    block: Block,
    properties_schema: PropertiesSchema,
    fs: FlowsheetBlock,
    block_ctx: "BlockContext",
) -> None:
    """
    Fix the properties of a block based on the properties schema.

    Args:
    - block: The block to fix the properties of.
    - properties_schema: The schema of the properties to fix.
    - fs: Used to store the properties in the properties map.
    """
    property_key: str
    property_info: PropertySchema
    for property_key, property_info in properties_schema.items():
        # Key is e.g "enth_mol"

        # TODO: Handle transformers
        # indexed_data = extract_indexes(
        #     property_info.data,
        #     property_info.unit,
        #     transformers,
        # )
        property_reference = collate_indexes(block, property_key)

        for property_value in property_info.data:
            discrete_indexes = property_value.discrete_indexes or []

            pv_id = property_value.id
            num_discrete_indexes = len(discrete_indexes)
            num_property_indexes = property_reference.index_set().dimen if property_reference.is_indexed() else 0
            num_continuous_indexes = num_property_indexes - num_discrete_indexes # This is the dimension of the property_value.value ndarray.

            # We have the convention that all the continuous indexes come first in property_reference, and then the discrete indexes.
            # This is what idaes normally does.
            
            
            # We need to get a slice to the current set of discrete indexes.
            if len(discrete_indexes) == 0:
                property_slice = property_reference # no indexes to worry about.
            else:
                property_slice = property_reference[tuple(
                    list(slice(None) for _ in range(num_continuous_indexes)) + discrete_indexes
                )]
            # Now property_slice is only indexed by the continuous indexes.
            
            # get a reference to the variable/expression. This will also add an index [None] if it is not indexed at all, i.e no continuous indexes.
            variable_references = Reference(property_slice, ctype=IndexedComponent) #Ctype=IndexedComponent avoids problems with DerivativeVar
            
            add_to_property_map(variable_references, pv_id, fs)
            
            # Because both pyomo and numpy flatten arrays with the last index changing fastest, we can just flatten both the index set and the values, and then iterate through them together.
            variable_indexes = list(variable_references.index_set())

            if (len(variable_indexes) == 0):
                print("Warning: No variables found for {property_key} with indexes {discrete_indexes}. This may be expected in the milk property package, which doesn't have a gas phase.")
                continue
            
            variable_transformed = variable_references
            

            if property_value.value is not None:
                variable_values = np.array([property_value.value]).flatten() # we put the value in an array to handle the case where it is a scalar.
                variable_values_with_units = [attach_unit(v, property_info.unit) for v in variable_values]
                variable_values_converted = [
                    units.convert(v, get_attached_unit(var)) for v, var in zip(variable_values_with_units, variable_references.values())
                ]

                #value = units.convert(property_value.value, get_attached_unit(var))
                if property_value.constraint is not None:
                    # add the constraint to the list of constraints to be added
                    # at the flowsheet level. The var value should also
                    # be a guess to maintain the degrees of freedom.
                    expr = property_value.constraint
                    fs.constraint_exprs.append((variable_transformed, expr, pv_id))
                    if property_value.controlled:
                        # this is a set point. To maintain degrees of freedom,
                        # use the manipulated variable as a guess during initialisation.
                        # this is a weird case because we might have guesses for both the
                        # manipulated variable and this variable, and we want to end up
                        # using the formula/constraint. If we were to try use this guess
                        # by eliminating the manipulated variable guess, we would run into
                        # degrees of freedom issues at the flowsheet level if elimination
                        # failed (since it then adds another flowsheet level constraint).
                        load_initial_guesses(variable_references, variable_values_converted)
                    else:
                        # not a set point, use this guess during initialization.
                        component = fix_slice(variable_references, variable_values_converted)
                        fs.guess_vars.append(component)
                elif property_value.controlled is not None:
                    # this value is a controlling variable, so don't fix it now
                    # it should be fixed after initialisation
                    block_ctx.add_controlled_var(variable_references, variable_values_converted, pv_id, property_value.controlled)
                elif property_value.guess:
                    block_ctx.add_guess_var( variable_references, variable_values_converted, pv_id)
                else:
                    components = fix_slice(variable_references, variable_values_converted)
                    # add_corresponding_constraint used to take the constraint returned by constrain_component if you are constraining an expression. TODO: Do we need to add this back in?
                    # from c = fix_var()
                    add_corresponding_constraint(fs, components, pv_id)



from ahuora_builder.methods.BlockContext import BlockContext