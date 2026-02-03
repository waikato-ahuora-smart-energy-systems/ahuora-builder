from .methods.load_unit_model import add_unit_model_to_flowsheet
from .methods.adapter_library import UnitModelConstructor, AdapterLibrary
from .flowsheet_manager_type import FlowsheetManager
from ahuora_builder_types import UnitModelId, UnitModelSchema
from idaes.core import UnitModelBlock



class UnitModelManager:
    """
    Manages the unit models in the flowsheet
    """

    def __init__(self, flowsheet_manager: FlowsheetManager) -> None:
        """
        Create a new unit model manager
        """
        self.flowsheet_manager = flowsheet_manager
        self._unit_models: dict[UnitModelId, UnitModelBlock] = {}

    def load(self) -> None:
        """
        load all unit models from the schema into the flowsheet
        """
        schema = self.flowsheet_manager.schema

        for unit_model_def in schema.unit_models:
            unit_model_type = unit_model_def.type
            if unit_model_type not in AdapterLibrary:
                raise Exception(
                    f"Unit model type '{unit_model_type}' is not in the adapter library."
                )
            adapter_constructor: UnitModelConstructor = AdapterLibrary[unit_model_type]
            self.load_from_def(unit_model_def, adapter_constructor)

    def load_from_def(
        self, unit_model_def: UnitModelSchema, adapter_constructor: UnitModelConstructor
    ) -> None:
        """
        Deserialise a unit model from a JSON object into a unit model adapter
        """
        # add the unit model to the flowsheet
        unit_model =  add_unit_model_to_flowsheet(
            unit_model_def, adapter_constructor, self.flowsheet_manager
        )
        
        # store a reference to the adapter
        self._unit_models[unit_model_def.id] = unit_model
