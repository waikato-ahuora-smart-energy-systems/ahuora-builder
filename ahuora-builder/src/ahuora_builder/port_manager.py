from pyomo.network import Port
from ahuora_builder_types import PortId


class PortManager:
    
    def __init__(self) -> None:
        self._map: dict[PortId, Port] = {}
    

    def register_port(self, id: PortId, port: Port) -> None:
        """Registers a port with the port map, so that arcs can connect to it by id"""
        self._map[id] =  port
    
    
    def get_port(self, id: PortId) -> Port:
        try:
            return self._map[id]
        except KeyError:
            raise KeyError(f"Port with id `{id}` not found")