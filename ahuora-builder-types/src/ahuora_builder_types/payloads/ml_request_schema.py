    
from typing import Optional
from pydantic import BaseModel

class MLTrainRequestPayload(BaseModel):
    """request schema"""
    datapoints: list
    columns: list[str]
    input_labels: list[str]
    output_labels: list[str]
    task_id: int

class MLTrainingCompletionPayload(BaseModel):
    """response schema"""
    json_response: dict
    error: Optional[str]  # error message if applicable
    log: Optional[str]  # log
    traceback: Optional[str]  # traceback if applicable
    task_id: int
    status: str
    
