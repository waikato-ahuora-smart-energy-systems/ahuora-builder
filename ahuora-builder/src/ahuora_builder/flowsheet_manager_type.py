# Because there are a lot of things that reference the flowsheet manager, we
# get a circular dependency.
# But this is only a problem for the type hints.
# so instead, we will use this trick:

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .flowsheet_manager import FlowsheetManager
else:
    FlowsheetManager = "FlowsheetManager"
# This only imports the type if we are checking types.
# Then as long as we use the type hints as a forward reference, we are okay.
# https://peps.python.org/pep-0484/#forward-references
# e.g
# from .flowsheet_manager_type import FlowsheetManager
# def __init__(self, flowsheet_manager: "FlowsheetManager"):
# rather than
# from .flowsheet_manager import FlowsheetManager
# def __init__(self, flowsheet_manager: FlowsheetManager):
#