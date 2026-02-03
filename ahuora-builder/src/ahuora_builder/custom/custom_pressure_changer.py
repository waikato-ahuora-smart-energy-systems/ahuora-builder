from idaes.models.unit_models.pressure_changer import PressureChangerData
from .inverted import add_inverted, initialise_inverted
from idaes.core import declare_process_block_class

@declare_process_block_class("CustomPressureChanger")
class CustomPressureChangerData(PressureChangerData):
    """
    Custom Pressure Changer model that includes inverted deltaP property.
    """

    def build(self, *args, **kwargs):
        """
        Build method for the CustomPressureChangerData class.
        This method initializes the control volume and sets up the model.
        """
        super().build(*args, **kwargs)

        # add deltaP_inverted as a property
        add_inverted(self, "deltaP")

    def initialize_build(
        self,*args,**kwargs,
    ):
        """
        Initialization method for the CustomPressureChangerData class.

        Args:
            state_args (dict): Arguments to be passed to the state block
            solver (str): Solver to use for initialization
            optarg (dict): Solver arguments dictionary
        """
        initialise_inverted(self, "deltaP")

        super().initialize_build(*args, **kwargs)
