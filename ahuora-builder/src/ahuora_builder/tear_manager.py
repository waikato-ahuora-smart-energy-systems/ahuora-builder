from pyomo.environ import value
from pyomo.network import Arc
from idaes.core.util.tables import _get_state_from_port
from pyomo.core.base.expression import ScalarExpression
from ahuora_builder_types.arc_schema import TearGuessSchema
from .flowsheet_manager_type import FlowsheetManager
from ahuora_builder_types import PortId
from .methods.adapter import fix_var


class TearManager:
    """
    Manages the tears in the flowsheet
    """

    def __init__(self, flowsheet_manager: FlowsheetManager):
        """
        Create a new tear manager
        """
        self._flowsheet_manager = flowsheet_manager
        self._tears: list[Arc] = []

    def load(self):
        """
        Load all the tears (from the recycle unitops)
        """
        schema = self._flowsheet_manager.schema

        for arc_schema in schema.arcs:
            if arc_schema.tear_guess:
                self.add_tear(arc_schema)

    def add_tear(self, arc_schema):
        """
        Add a tear to the flowsheet
        """
        portId = arc_schema.destination
        guess = arc_schema.tear_guess

        port = self._flowsheet_manager.ports.get_port(portId)
        arc = port.arcs()[0]
        self._tears.append(arc)

        """
        During model loading, we add in all the constraints and fix the variables
        for this port. Since it is a tear, things need to be deactivated/unfixed
        to ensure 0 degrees of freedom

        If the state block is defined by constraints, we need to solve it to get the
        correct values for the state variables. Then we deactivate the constraints
        since we will be fixing the state variables instead (where applicable).
        """
        
        # hardcoding time indexes to [0] for now
        time_indexes = [0]

        # need to get the correct value for state variables before running
        # sequential decomposition (since all state variables are fixed
        # during sequential decomposition).
        sb = _get_state_from_port(port, time_indexes[0])
        # deactivate any guesses
        for key, value in guess.items():
            var = getattr(sb, key)
            if value != True:
                if isinstance(var,ScalarExpression):
                    pass # we want to solve with constraints
                else:
                    # this might give too few dof if we have other constraints. We need to unfix
                    # this value
                    var.unfix()

        if len(list(sb.constraints.component_objects())) > 0:
            # state block is defined by some constraints, so we need to solve it.
            # initialize the state block, which should put the correct value in
            # the state variables
            sb.parent_component().initialize()
            # deactivate the state block constraints, since we should use the
            # state variables as guesses (or fix them instead)
            sb.constraints.deactivate()

        blk = sb.parent_component()
        for key in sb.define_state_vars():
            if guess.get(key, False):
                # deactivate the equality constraint (expanded arcs)
                expanded_arc = getattr(
                    self._flowsheet_manager.model.fs, arc._name + "_expanded"
                )
                equality_constraint = getattr(expanded_arc, key + "_equality")
                equality_constraint.deactivate()

                # fix the variable
                for b in blk.values():
                    getattr(b, key).fix()

            else:
                # unfix this variable
                for b in blk.values():
                    getattr(b, key).unfix()
