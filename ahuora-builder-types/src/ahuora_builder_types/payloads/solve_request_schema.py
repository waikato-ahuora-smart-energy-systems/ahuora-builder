from typing import Optional

from pydantic import BaseModel
from ahuora_builder_types import FlowsheetSchema
from ahuora_builder_types.flowsheet_schema import SolvedFlowsheetSchema


class IdaesSolveRequestPayload(BaseModel):
    """
    Payload for IDAES solve requests
    """
    flowsheet: FlowsheetSchema
    solve_index: Optional[int] = None
    scenario_id: Optional[int] = None
    task_id: int
    perform_diagnostics: bool = False


class IdaesSolveCompletionPayload(BaseModel):
    """
    Payload for IDAES solve completion events
    """
    flowsheet: Optional[SolvedFlowsheetSchema]
    input_flowsheet: FlowsheetSchema
    solve_index: Optional[int]
    scenario_id: Optional[int] = None
    task_id: int
    status: str
    error: Optional[dict]
    timing: dict
    log: str
    traceback: Optional[str]
    diagnostics_raw_text: Optional[str] = None

class MultiSolvePayload(BaseModel):
    """
    Payload for dispatching multi-steady state solves
    """
    task_id: int
    scenario_id: int
