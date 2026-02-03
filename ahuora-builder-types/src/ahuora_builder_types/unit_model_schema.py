from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, Union, Optional, Literal, NewType

from typing_extensions import TypeAliasType

from .id_types import UnitModelId, PortId, PropertyValueId

class SolvedPropertyValueSchema(BaseModel):
    id: PropertyValueId
    name: str
    value: float | list[float] # Dynamics will have a list of values, rather than a single value.
    unit: str
    unknown_units: bool = False

class PropertyValueSchema(BaseModel):
    id: PropertyValueId
    discrete_indexes: Optional[list[str]] = None
    value: Optional[float | list[float]] = None
    controlled: Optional[int] = None
    guess: bool = False
    constraint: Optional[str] = None

class PropertySchema(BaseModel):
    data: list[PropertyValueSchema]
    unit: Optional[str] = ""


PropertiesSchema = Dict[str, "PropertySchema"]


class PortSchema(BaseModel):
    id: PortId
    properties: PropertiesSchema


PortsSchema = Dict[str, PortSchema]


class UnitModelSchema(BaseModel):
    id: UnitModelId
    type: str  # eg. "pump", "heater", "mixer"
    name: str
    args: dict[str, Any]
    properties: PropertiesSchema
    ports: PortsSchema
    initial_values: Optional[dict[str, dict]] = Field(default_factory=dict) # IDAES initial values from their to_json method