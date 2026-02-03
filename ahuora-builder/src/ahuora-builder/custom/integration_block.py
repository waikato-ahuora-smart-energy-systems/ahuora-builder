#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Base class for unit models
"""

from pyomo.environ import check_optimal_termination
from pyomo.common.config import ConfigValue
from pyomo.network import Port

from idaes.core.base.process_base import (
    declare_process_block_class,
    ProcessBlockData,
    useDefault,
)
from idaes.core.base.property_base import StateBlock
from idaes.core.base.control_volume_base import (
    ControlVolumeBlockData,
    FlowDirection,
    MaterialBalanceType,
)
from idaes.core.util.exceptions import (
    BurntToast,
    ConfigurationError,
    BalanceTypeNotSupportedError,
    InitializationError,
)
from idaes.core.util.tables import create_stream_table_dataframe
import idaes.logger as idaeslog
from idaes.core.solvers import get_solver
from idaes.core.util.config import DefaultBool
from idaes.core.initialization import SingleControlVolumeUnitInitializer
from pyomo.environ import Var, units, Expression
from pyomo.dae import Integral


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("IntegrationBlock")
class IntegrationBlockData(ProcessBlockData):
    """
    This block has no inlets or outlets, but one value that you can get the integral for.
    """

    # Create Class ConfigBlock
    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            default=useDefault,
            domain=DefaultBool,
            description="Dynamic model flag",
            doc="""Indicates whether this model will be dynamic or not,
**default** = useDefault.
**Valid values:** {
**useDefault** - get flag from parent (default = False),
**True** - set as a dynamic model,
**False** - set as a steady-state model.}""",
        ),
    )

    def build(self):
        """
        
        """
        super(IntegrationBlockData, self).build()

        self.variable = Var(
            self.flowsheet().time
        )

        #self.time = self.flowsheet().time

        
        
        def _integral(m):
            time_steps = sorted(self.flowsheet().time)
            value = m.variable
            return sum([
                # Trapezium rule
                # average the value at the current and next time step, then multiply by the time difference
                0.5 * (value[time_steps[i]] + value[time_steps[i+1]]) * (time_steps[i+1] - time_steps[i]) 
                for i in range(len(time_steps)-1)
            ])
        
        self.integral = Expression(
            rule=_integral
        )
        
        
        # if self.config.dynamic:
        #     # create a variable to hold the normal value, and one to hold the integral value
        # else:

