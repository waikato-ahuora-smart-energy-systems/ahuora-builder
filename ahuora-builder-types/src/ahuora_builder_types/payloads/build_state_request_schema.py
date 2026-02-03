# request types for the build_state endpoint
from typing import Optional
from pydantic import BaseModel
from ahuora_builder_types.flowsheet_schema import PropertyPackageType
from ahuora_builder_types.unit_model_schema import PropertySchema, SolvedPropertyValueSchema


class BuildStateRequestSchema(BaseModel):
    """build_state request schema"""
    property_package: PropertyPackageType  # property package to use
    properties: dict[str, PropertySchema]  # properties to initialize the state block with

class BuildStateResponseSchema(BaseModel):
    """build_state response schema"""
    properties: Optional[list[SolvedPropertyValueSchema]]  # properties in the state block
    error: Optional[str]  # error message if applicable
    log: Optional[str]  # log
    traceback: Optional[str]  # traceback if applicable