from pydantic import BaseModel, Field
from typing import Optional
from .id_types import PortId, ArcId


class TearGuessSchema(BaseModel):
    """
    Indicates whether these properties should be fixed
    at the tear stream.
    """
    flow_mol: bool
    pressure: bool
    enth_mol: bool


class ArcSchema(BaseModel):
    """
    Stores the schema required on a JSON object to define an arc in IDAES.

    id: The id of the arc, for identification of the return arc.
    source_id: The id of the source port. Each port has a unique id.
    destination_id: The id of the destination port. Each port has a unique id.
    tear_guess: A dictionary mapping variables to whether they should be fixed at the tear stream.

    All fields are required.
    """
    id: ArcId
    source: PortId
    destination: PortId
    tear_guess: Optional[dict[str, bool]] = None