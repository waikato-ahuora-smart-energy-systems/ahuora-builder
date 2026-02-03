from ahuora_builder.methods.adapter import add_corresponding_constraint, fix_var,fix_slice, load_initial_guess


from idaes.core import FlowsheetBlock
from pyomo.core.base.constraint import Constraint
from pyomo.environ import Block, Component, Reference
from ahuora_builder_types.id_types import PropertyValueId
from pyomo.core.base.indexed_component_slice import (
    IndexedComponent_slice,
    _IndexedComponent_slice_iter,
)
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent


class BlockContext:
    """
    Where possible, we want to fix variables at the block level (ie. unit model, state block)
    rather than at the flowsheet level. This is because it is easier to solve a smaller model
    during initialization, rather than dumping complexity on the solver when solving the entire
    flowsheet.

    Each controlling variable in the model is accompanied by a guess variable. Normally, the
    guess variable is fixed during initialization, and unfixed after, while the controlling
    variable (set point) is a constraint at the flowsheet level. However, if both the guess and
    controlling variable are on the same block, we can avoid this and fix the controlling variable
    directly at the block level.

    This class provides a context to store controlling variables and guess variables while fixing
    within a block. We can then apply a simple heuristic to eliminate as many pairs of guess and
    controlling variables as possible.
    """

    def __init__(self, flowsheet: FlowsheetBlock):
        """
        - blk: Pyomo block the Var is on (can add constraints to this block)
        - var: Pyomo Var to fix/constrain
        - value: value to fix/constrain the Var to
        - id: id of the property, to store the created Constraint in the properties map
        """
        # property id: ( var_reference, values)
        self._guess_vars: dict[PropertyValueId, tuple[ IndexedComponent | IndexedComponent_slice, list[float]]] = {}
        # property id: ( var_reference, values, guess_id)
        self._controlled_vars: dict[PropertyValueId, tuple[IndexedComponent | IndexedComponent_slice, list[float], PropertyValueId]] = {}
        self._flowsheet = flowsheet

    def add_guess_var(self, var_references : IndexedComponent | IndexedComponent_slice, values : list[float], propertyvalue_id : PropertyValueId):
        self._guess_vars[propertyvalue_id] = ( var_references, values)

    def add_controlled_var(self, var_references: IndexedComponent | IndexedComponent_slice, values: list[float], propertyvalue_id: PropertyValueId, guess_propertyvalue_id: PropertyValueId):
        self._controlled_vars[propertyvalue_id] = (var_references, values, guess_propertyvalue_id)

    def apply_elimination(self):
        """
        Try to eliminate as many guess vars/flowsheet-level constraints as possible.
        Fix the remaining guess vars or add the remaining controlled vars as constraints.
        """
        # TODO: Update apply_elimination with the lists of values now.
        fs = self._flowsheet
        for id, (var_refs, values, guess_id) in self._controlled_vars.items():
            # see if we can eliminate this controlled var
            if guess_id in self._guess_vars:
                # fix the controlled var
                c = fix_slice(var_refs, values)
                add_corresponding_constraint(fs, c, id)
                # load the initial guess for the guess var
                var_refs, values = self._guess_vars[guess_id]
                load_initial_guess(var_refs, values)
                # eliminate the guess var
                del self._guess_vars[guess_id]
            else:
                # add the control as a flowsheet-level constraint
                # As the values are flattended into a list, we also need to flatten the index set into a list.
                var_refs_list = list(var_refs.values()) # returns a list of VarData or ExpressionData objects
                def constraint_rule(blk,idx):
                    return var_refs_list[idx] == values[idx]
                c = Constraint(range(len(var_refs_list)), rule=constraint_rule)
                name = f"control_constraint_{id}" # Maybe we could use the var name or something here? but it's a bit harder with indexed constraints. remember it has to be unique!
                self._flowsheet.add_component(name, c)
                add_corresponding_constraint(fs, c, id)

        # fix the remaining guess vars
        for id, (var_refs, values) in self._guess_vars.items():
            c = fix_slice(var_refs, values)
            self._flowsheet.guess_vars.append(c)