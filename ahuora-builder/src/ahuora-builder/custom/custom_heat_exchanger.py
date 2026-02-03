

# Import Pyomo libraries
from pyomo.environ import (
    Block,
    Var,
    Param,
    log,
    Reference,
    PositiveReals,
    ExternalFunction,
    units as pyunits,
    check_optimal_termination,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
)

import idaes.logger as idaeslog
from idaes.core.util.functions import functions_lib
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.models.unit_models.heater import (
    _make_heater_config_block,
    _make_heater_control_volume,
)

from idaes.core.util.misc import add_object_reference
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.initialization import SingleControlVolumeUnitInitializer
from idaes.models.unit_models.heat_exchanger import HX0DInitializer, _make_heat_exchanger_config, HeatExchangerData
from .inverted import add_inverted, initialise_inverted
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("CustomHeatExchanger", doc="Simple 0D heat exchanger model.")
class CustomHeatExchangerData(HeatExchangerData):

    def build(self,*args,**kwargs) -> None:
        """
        Begin building model.
        """
        super().build(*args,**kwargs)
        # Add an inverted DeltaP
        add_inverted(self.hot_side, "deltaP")
        add_inverted(self.cold_side, "deltaP")

    def initialize_build(
        self,
        state_args_1=None,
        state_args_2=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
        duty=None,
    ):
        """
        Heat exchanger initialization method.

        Args:
            state_args_1 : a dict of arguments to be passed to the property
                initialization for the hot side (see documentation of the specific
                property package) (default = {}).
            state_args_2 : a dict of arguments to be passed to the property
                initialization for the cold side (see documentation of the specific
                property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)
            duty : an initial guess for the amount of heat transferred. This
                should be a tuple in the form (value, units), (default
                = (1000 J/s))

        Returns:
            None

        """
        # So, when solving with a correct area, there can be problems
        # That's because if the area's even slightly too large, it becomes infeasible
        if not self.area.fixed:
            self.area.value = self.area.value * 0.8

        initialise_inverted(self.hot_side, "deltaP")
        initialise_inverted(self.cold_side, "deltaP")

        # Set solver options
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        flags1 = self.hot_side.initialize(
            outlvl=outlvl, optarg=optarg, solver=solver, state_args=state_args_1
        )

        init_log.info_high("Initialization Step 1a (hot side) Complete.")

        flags2 = self.cold_side.initialize(
            outlvl=outlvl, optarg=optarg, solver=solver, state_args=state_args_2
        )
        init_log.info_high("Initialization Step 1b (cold side) Complete.")
        # ---------------------------------------------------------------------
        # Solve unit without heat transfer equation
        self.heat_transfer_equation.deactivate()
        if hasattr( self.cold_side.properties_out[0], "constraints"):
            self.cold_side.properties_out[0].constraints.deactivate()
        if hasattr( self.hot_side.properties_out[0], "constraints"):
            self.hot_side.properties_out[0].constraints.deactivate()

        # Get side 1 and side 2 heat units, and convert duty as needed
        s1_units = self.hot_side.heat.get_units()
        s2_units = self.cold_side.heat.get_units()

        # Check to see if heat duty is fixed
        # WE will assume that if the first point is fixed, it is fixed at all points
        if not self.cold_side.heat[self.flowsheet().time.first()].fixed:
            cs_fixed = False
            if duty is None:
                # Assume 1000 J/s and check for unitless properties
                if s1_units is None and s2_units is None:
                    # Backwards compatibility for unitless properties
                    s1_duty = -1000
                    s2_duty = 1000
                else:
                    s1_duty = pyunits.convert_value(
                        -1000, from_units=pyunits.W, to_units=s1_units
                    )
                    s2_duty = pyunits.convert_value(
                        1000, from_units=pyunits.W, to_units=s2_units
                    )
            else:
                # Duty provided with explicit units
                s1_duty = -pyunits.convert_value(
                    duty[0], from_units=duty[1], to_units=s1_units
                )
                s2_duty = pyunits.convert_value(
                    duty[0], from_units=duty[1], to_units=s2_units
                )

            self.cold_side.heat.fix(s2_duty)
            for i in self.hot_side.heat:
                self.hot_side.heat[i].value = s1_duty
        else:
            cs_fixed = True
            for i in self.hot_side.heat:
                self.hot_side.heat[i].set_value(self.cold_side.heat[i])
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))
        if not cs_fixed:
            self.cold_side.heat.unfix()
        if hasattr( self.cold_side.properties_out[0], "constraints"):
            self.cold_side.properties_out[0].constraints.activate()
        if hasattr( self.hot_side.properties_out[0], "constraints"):
            self.hot_side.properties_out[0].constraints.activate()
        self.heat_transfer_equation.activate()

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        # ---------------------------------------------------------------------

        # Release Inlet state
        self.hot_side.release_state(flags1, outlvl=outlvl)
        self.cold_side.release_state(flags2, outlvl=outlvl)

        init_log.info("Initialization Completed, {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )