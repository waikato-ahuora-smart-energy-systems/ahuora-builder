from ..properties_manager import PropertiesManager
from typing import Iterator, TextIO
import sys
import pyomo.environ as pyo
from pyomo.core.base.constraint import ScalarConstraint

def compute_infeasibilities(properties_map: PropertiesManager) -> Iterator[tuple[any,float]]:
    """
    Compute how far each constrained property is from its target value.
    
    :param properties_map: PropertiesManager instance containing property components
    :type properties_map: PropertiesManager
    :return: A tuple list of property IDs and their infeasibility magnitudes
    :rtype: list[tuple[Any, float]]
    """

    for id, property_component in properties_map.items():
        if property_component.corresponding_constraint is not None and type(next(iter(property_component.corresponding_constraint))) is ScalarConstraint:
            # Assuming the component is a Pyomo Var or Expression
            current_value :float = pyo.value(next(iter(property_component.corresponding_constraint)))
            target_value :float = pyo.value(next(iter(property_component.corresponding_constraint)).upper)
            infeasibility = abs(current_value - target_value)
            yield (id, infeasibility)

def print_infeasibilities(properties_map: PropertiesManager):
    """
    Print the infeasibilities of constrained properties, sorted by magnitude.
    
    :param properties_map: PropertiesManager instance containing property components
    :type properties_map: PropertiesManager
    """


    # Sort from most to least infeasible
    infeasibilities = list(compute_infeasibilities(properties_map))
    infeasibilities.sort(key=lambda x: x[1], reverse=True)

    print("Property Infeasibilities:")
    for id, infeasibility in infeasibilities:
        print(f"Property: {properties_map.get(id).name}, Infeasibility: {infeasibility}")