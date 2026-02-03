from typing import Any, Callable
from abc import ABC, abstractmethod
from idaes.core.base.property_base import PhysicalParameterBlock
from idaes.core.base.reaction_base import ReactionParameterBlock
from ..flowsheet_manager_type import FlowsheetManager
from ahuora_builder_types import PropertyPackageId




class AdapterBase(ABC):
    """
    Base class for adapter methods
    """

    @abstractmethod
    def run(self, value, flowsheet_manager: FlowsheetManager) -> Any:
        pass


class PropertyPackage(AdapterBase):
    """Get a property package by ID"""

    def run(self, id: PropertyPackageId, flowsheet_manager: FlowsheetManager) -> PhysicalParameterBlock:
        return flowsheet_manager.property_packages.get(id)

class ReactionPackage(AdapterBase):
    """Get a reaction package"""
    def run(self, id: str, flowsheet_manager: FlowsheetManager) -> ReactionParameterBlock:
        return flowsheet_manager.model.fs.reaction_params
    
class ReactorPropertyPackage(AdapterBase):
     def run(self,value,flowsheet_manager) -> PhysicalParameterBlock:
        return flowsheet_manager.model.fs.peng_robinson   


class Value(AdapterBase):
    """Do not modify the value passed"""

    def run(self, value: Any,flowsheet_manager) -> Any:
        return value


class Constant(AdapterBase):
    """Always return the argument passed"""
    def __init__(self,constant) -> None:
        self._constant = constant

    def run(self, value, flowsheet_manager) -> Callable:
        return self._constant


class Dictionary(AdapterBase):
    """Validate a dictionary against a schema and adapt the values"""
    def __init__(self,schema) -> None:
        self._schema = schema

    def run(self, dictionary: dict, flowsheet_manager) -> Callable[[dict[str, Callable]], dict]:
        schema = self._schema

        result = {}
        for item, adapter in schema.items():
            # TODO: do we need to handle optional items better?
            result[item] = adapter.run(dictionary.get(item, None),flowsheet_manager)
        return result


class PowerPropertyPackage(AdapterBase):
    def run(self,value,flowsheet_manager) -> PhysicalParameterBlock:
        return flowsheet_manager.model.fs.power_pp # See __init__ method of Flowsheet_Manager.py 

class acPropertyPackage(AdapterBase):
    def run(self,value,flowsheet_manager) -> PhysicalParameterBlock:
        return flowsheet_manager.model.fs.ac_pp # See __init__ method of Flowsheet_Manager.py 

class TransformerPropertyPackage(AdapterBase):
    def run(self,value,flowsheet_manager) -> PhysicalParameterBlock:
        return flowsheet_manager.model.fs.tr_pp # See __init__ method of Flowsheet_Manager.py 


