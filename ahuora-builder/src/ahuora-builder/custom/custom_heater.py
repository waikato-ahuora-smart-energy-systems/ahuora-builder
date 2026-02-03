from idaes.core import declare_process_block_class
from idaes.models.unit_models.heater import HeaterData
from .add_initial_dynamics import add_initial_dynamics
from .inverted import add_inverted, initialise_inverted, disable_inverted, enable_inverted

@declare_process_block_class("DynamicHeater")
class DynamicHeaterData(HeaterData):
    """
    Dynamic Heater unit model class.
    This extends the Heater class to include reference variables for initial holdup and initial accumulation. 
    Which makes it easier for us to set initial conditions in the frontend.
    """

    def build(self,*args, **kwargs):
        """
        Build method for the DynamicHeaterData class.
        This method initializes the control volume and sets up the model.
        """
        super().build(*args, **kwargs)

        if hasattr(self,"deltaP"):
            # else has_pressure_change is false
            add_inverted(self, "deltaP")

        add_initial_dynamics(self)
    
    def initialize_build(
        self,*args,**kwargs,
    ):
        """
        Initialize method for the DynamicHeaterData class.
        This method initializes the control volume and sets up the model.
        """
        if hasattr(self,"deltaP"):
            initialise_inverted(self, "deltaP")
            disable_inverted(self, "deltaP")

        super().initialize_build(*args, **kwargs)

        if hasattr(self,"deltaP"):
            enable_inverted(self, "deltaP")
