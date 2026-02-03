from typing import Optional
from pydantic import BaseModel
from ahuora_builder_types import FlowsheetSchema
from ahuora_builder_types.flowsheet_schema import SolvedFlowsheetSchema
from ahuora_builder_types.payloads.solve_request_schema import IdaesSolveRequestPayload
from .flowsheet_manager import FlowsheetManager


class SolveModelResult(BaseModel):
    input_flowsheet: FlowsheetSchema
    output_flowsheet: SolvedFlowsheetSchema
    solve_index: Optional[int] = None
    scenario_id: Optional[int] = None
    task_id: int
    timing: dict

def solve_model(solve_request: IdaesSolveRequestPayload) -> SolveModelResult:
    """Solves the model and returns the results"""
    flowsheet = FlowsheetManager(solve_request.flowsheet)
    #print(solve_request.flowsheet.model_dump_json())
    flowsheet.load()
    flowsheet.initialise()
    flowsheet.report_statistics()
    if solve_request.perform_diagnostics:
        flowsheet.diagnose_problems()
    flowsheet.check_model_valid()
    flowsheet.solve()
    flowsheet.optimize()
    result = flowsheet.serialise()

    return SolveModelResult(
        input_flowsheet=solve_request.flowsheet,
        output_flowsheet=result,
        solve_index=solve_request.solve_index,
        scenario_id=solve_request.scenario_id,
        task_id=solve_request.task_id,
        timing=flowsheet.timing.close()
    )