from pyomo.network import Arc
from .flowsheet_manager_type import FlowsheetManager
from ahuora_builder_types import PortId


class ArcManager:

    def __init__(self, flowsheet_manager: FlowsheetManager):
        """
        Initializes the arc manager
        """
        self._flowsheet_manager = flowsheet_manager

    def load(self):
        """
        Loads arcs from the schema and adds them to the flowsheet
        """
        schema = self._flowsheet_manager.schema

        for arc_schema in schema.arcs:
            self.add_arc(arc_schema.source, arc_schema.destination)

    def add_arc(self, from_port_id: PortId, to_port_id: PortId):
        """
        Adds an arc between two ports
        """
        port_manager = self._flowsheet_manager.ports
        from_port = port_manager.get_port(from_port_id)
        to_port = port_manager.get_port(to_port_id)
        arc = Arc(source=from_port, destination=to_port)
        self._flowsheet_manager.model.fs.add_component(
            f"arc_{from_port_id}_{to_port_id}", arc
        )
