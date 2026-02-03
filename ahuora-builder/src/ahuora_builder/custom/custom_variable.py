from idaes.models.control.controller import PIDControllerData, ControllerType, ControllerMVBoundType, ControllerAntiwindupType, smooth_bound, smooth_heaviside
from pyomo.environ import Var
from idaes.core import UnitModelBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue, In, Bool
import pyomo.environ as pyo
from idaes.core.util.exceptions import ConfigurationError
import pyomo.dae as pyodae
from idaes.core.util import scaling as iscale
import functools


@declare_process_block_class(
    "CustomVariable",
    doc="Custom variable model block. You can finally be free!",
)
class CustomVariableData(UnitModelBlockData):

    CONFIG = UnitModelBlockData.CONFIG()

    def build(self):
        """
        Build the custom variable block
        """
        super().build()
        self.variable = Var(
            self.flowsheet().time,
            initialize=0.0,
            doc="A variable you are free to do whatever you want with",
        )
