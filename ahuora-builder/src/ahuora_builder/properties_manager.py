from ahuora_builder_types.id_types import PropertyValueId
from pyomo.core.base.indexed_component import IndexedComponent
class PropertyComponent:
    def __init__(self, name: str, component : IndexedComponent, unknown_units=False):
        """
        Args:
        - name (str): The name of the property component (for debugging purposes)
        - component (Component): The Pyomo component; i.e., Expression, Var
        - unknown_units (bool): If units are unknown, idaes_factory will do some
            additional processing to determine the unit category.
        - corresponding_constraint (Constraint | Var | None): The component
            (Var or Constraint) that was fixed or activated to set the value of
            this property. Given an id in the properties dictionary, this field
            can be used to unfix or deactivate the corresponding constraint.
            Defaults to None (allowed for properties that are not fixed).
        """
        self.name: str = name
        self.component: IndexedComponent = component
        self.unknown_units = unknown_units
        self.corresponding_constraint = None # TODO: Type


class PropertiesManager:
    def __init__(self):
        self.properties : dict[PropertyValueId,PropertyComponent] = {}

    def add(self, id: PropertyValueId, indexed_component: IndexedComponent, name : str, unknown_units=False)-> IndexedComponent:
        self.properties[id] = PropertyComponent(name, indexed_component, unknown_units)
    
    def get(self, id: PropertyValueId):
        return self.properties[id]
    
    def get_component(self, id: PropertyValueId):
        return self.get(id).component
    
    def get_constraint(self, id: PropertyValueId):
        return self.get(id).corresponding_constraint
    
    def add_constraint(self, id: PropertyValueId, constraint):
        # assumes the property has already been added
        self.get(id).corresponding_constraint = constraint
    
    def items(self):
        return self.properties.items()