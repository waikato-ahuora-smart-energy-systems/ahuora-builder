from typing import Optional, Literal
from pydantic import BaseModel, Field
from ahuora_builder_types import UnitModelSchema, ArcSchema


class UnfixedVariableSchema(BaseModel):
    id: int
    lower_bound: Optional[float]
    upper_bound: Optional[float]


class OptimizationSchema(BaseModel):
    objective: int # id of the property to optimize
    sense: Literal["minimize"] | Literal["maximize"]
    unfixed_variables: list[UnfixedVariableSchema]
