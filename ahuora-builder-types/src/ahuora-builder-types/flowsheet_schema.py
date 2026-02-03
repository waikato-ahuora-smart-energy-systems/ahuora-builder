from typing import Optional, Literal
from pydantic import BaseModel, Field
from ahuora_builder_types import UnitModelSchema, ArcSchema
from ahuora_builder_types.unit_model_schema import SolvedPropertyValueSchema
from ahuora_builder_types.scenario_schema import OptimizationSchema

class PropertyPackageType(BaseModel):
    id: int
    type: str
    compounds: list[str]
    phases: list[str]

def default_time_set():
    return list([0])

class FlowsheetSchema(BaseModel):
    """Represents the schema required on a JSON object to define a flowsheet in IDAES."""
    id: int
    name: str = Field(default="")
    description: str = Field(default="")
    dynamic: bool # If the flowsheet is dynamic or steady state
    is_rating_mode: bool = False # If the flowsheet is in rating mode
    time_set: Optional[list[float]] = Field(default_factory=lambda: default_time_set()) # Time set to use in idaes.
    property_packages: list[PropertyPackageType] # Property packages are not required in return type
    unit_models: list[UnitModelSchema]
    arcs: list[ArcSchema]
    machineLearningBlock: Optional[list[dict]] = None
    expressions: Optional[list[dict]] = None
    optimizations: Optional[list[OptimizationSchema]] = None # Convert to list[ScenarioSchema]
    disable_initialization: bool = False  # If the flowsheet should not be initialised
    solver_option: Optional[str] = "ipopt"


class SolvedFlowsheetSchema(BaseModel):
    id: int
    properties: list[SolvedPropertyValueSchema]
    initial_values: dict[str, dict]
