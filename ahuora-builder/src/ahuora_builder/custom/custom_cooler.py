from idaes.models.unit_models.heater import HeaterData
from .inverted import add_inverted, initialise_inverted, enable_inverted, disable_inverted
from idaes.core import declare_process_block_class

@declare_process_block_class("CustomCooler")
class CustomCoolerData(HeaterData):
    """
    Custom Cooler model that includes inverted deltaP and Heat Added properties.
    """

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # add deltaP_inverted as a property
        add_inverted(self, "heat_duty")
        if hasattr(self,"deltaP"):
            add_inverted(self, "deltaP")

    def initialize_build(
        self,*args,**kwargs,
    ):
        initialise_inverted(self, "heat_duty")
        disable_inverted(self, "heat_duty")

        if hasattr(self,"deltaP"):
            initialise_inverted(self, "deltaP")
            disable_inverted(self, "deltaP")

        super().initialize_build(*args, **kwargs)

        enable_inverted(self, "heat_duty")
        if hasattr(self,"deltaP"):
            enable_inverted(self, "deltaP")