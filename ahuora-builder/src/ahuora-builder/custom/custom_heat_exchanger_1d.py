from pyomo.environ import Var, Constraint, Reference, units as pyunits
from pyomo.common.numeric_types import value
from pyomo.opt.results.solver import check_optimal_termination
from idaes.core import declare_process_block_class
from idaes.core.util import scaling as iscale
from idaes.models.unit_models.heat_exchanger_1D import HeatExchanger1DData, HX1DInitializer
from idaes.models.unit_models.heat_exchanger import HeatExchangerFlowPattern
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import InitializationError
import idaes.logger as idaeslog

class CustomHX1DInitializer(HX1DInitializer):
    """
    Use our custom control-volume initialize (no source port-member fixing).
    """
    def initialize_control_volume(self, cv, state_args=None):
        return initialize(
            cv,
            state_args=state_args,
            outlvl=self.get_output_level(),
        )

@declare_process_block_class(
    "CustomHeatExchanger1D",
    doc="1D Heat Exchanger with overall U tied to local heat_transfer_coefficient.",
)
class CustomHeatExchanger1DData(HeatExchanger1DData):
    # Use our initializer so both sides use the custom CV initialize
    default_initializer = CustomHX1DInitializer

    CONFIG = HeatExchanger1DData.CONFIG()

    def build(self):
        super().build()
        # Ends of the tube along the length axis (start and end positions)
        x_first = self.hot_side.length_domain.first()
        x_last = self.hot_side.length_domain.last()

        # Hot side: inlet at start, outlet at end
        x_hot_in, x_hot_out = x_first, x_last

        # Cold side: depends on flow pattern
        if self.config.flow_type == HeatExchangerFlowPattern.cocurrent:
            x_cold_in, x_cold_out = x_first, x_last
        else:
            x_cold_in, x_cold_out = x_last, x_first

        # Time-only inlet/outlet views of the boundary states (no extra vars/cons)
        self.hot_side.properties_in = Reference(self.hot_side.properties[:, x_hot_in])
        self.hot_side.properties_out = Reference(self.hot_side.properties[:, x_hot_out])
        self.cold_side.properties_in = Reference(self.cold_side.properties[:, x_cold_in])
        self.cold_side.properties_out = Reference(self.cold_side.properties[:, x_cold_out])

        # Overall U
        self.overall_heat_transfer_coefficient = Var(
            self.flowsheet().time,
            initialize=500.0,
            bounds=(1.0, 1e5),
            units=pyunits.W / pyunits.m**2 / pyunits.K,
            doc="Overall (constant along length) heat transfer coefficient U.",
        )

        @self.Constraint(self.flowsheet().time, self.hot_side.length_domain)
        def overall_heat_transfer_coefficient_def(b, t, x):
            return b.overall_heat_transfer_coefficient[t] == b.heat_transfer_coefficient[t, x]

        iscale.set_scaling_factor(self.overall_heat_transfer_coefficient, 1e-3)

    def initialize_build(
        self,
        hot_side_state_args=None,
        cold_side_state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
        duty=None,
    ):
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        opt = get_solver(solver, optarg)

        # Sync length values
        if self.length.fixed:
            self.cold_side.length.set_value(self.length)
        elif self.cold_side.length.fixed:
            self.length.set_value(self.cold_side.length)

        # Initialize control volumes with length fixed
        Lfix = self.hot_side.length.fixed
        self.hot_side.length.fix()
        flags_hot_side = initialize(
            self.hot_side,
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=hot_side_state_args,
        )
        if not Lfix:
            self.hot_side.length.unfix()

        Lfix = self.cold_side.length.fixed
        self.cold_side.length.fix()
        # Use our custom CV initialize here as well
        flags_cold_side = initialize(
            self.cold_side,
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=cold_side_state_args,
        )
        if not Lfix:
            self.cold_side.length.unfix()

        init_log.info_high("Initialization Step 1 Complete.")

        # Fixed-duty solve
        hot_units = self.hot_side.config.property_package.get_metadata().get_derived_units
        cold_units = self.cold_side.config.property_package.get_metadata().get_derived_units
        t0 = self.flowsheet().time.first()

        # Use inlet indices for each side
        x_hot_in = self.hot_side.length_domain.first()
        x_cold_in = self.cold_side.length_domain.first() if self.config.flow_type == HeatExchangerFlowPattern.cocurrent else self.cold_side.length_domain.last()

        if duty is None:
            duty = value(
                0.25
                * self.heat_transfer_coefficient[t0, x_hot_in]
                * self.area
                * (
                    self.hot_side.properties[t0, x_hot_in].temperature
                    - pyunits.convert(
                        self.cold_side.properties[t0, x_cold_in].temperature,
                        to_units=hot_units("temperature"),
                    )
                )
            )
        else:
            duty = pyunits.convert_value(duty[0], from_units=duty[1], to_units=hot_units("power"))

        duty_per_length = value(duty / self.length)

        # Fix heat duties
        for v in self.hot_side.heat.values():
            v.fix(-duty_per_length)
        for v in self.cold_side.heat.values():
            v.fix(pyunits.convert_value(duty_per_length, to_units=cold_units("power")/cold_units("length"), from_units=hot_units("power")/hot_units("length")))

        # Deactivate heat duty constraints and solve
        self.heat_transfer_eq.deactivate()
        self.heat_conservation.deactivate()
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        # Unfix heat duties and re-activate constraints
        for v in self.hot_side.heat.values():
            v.unfix()
        for v in self.cold_side.heat.values():
            v.unfix()
        self.heat_transfer_eq.activate()
        self.heat_conservation.activate()
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))

        release_state(self.hot_side, flags_hot_side)
        release_state(self.cold_side, flags_cold_side)

        if res is not None and not check_optimal_termination(res):
            raise InitializationError(f"{self.name} failed to initialize successfully. See logs.")

        init_log.info("Initialization Complete.")


def initialize(
        blk,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        optarg=None,
        solver=None,
        hold_state=True,
    ):
        """
        Initialization routine for 1D control volume.

        Keyword Arguments:
            state_args: a dict of arguments to be passed to the property
                package(s) to provide an initial state for initialization
                (see documentation of the specific property package) (default = {}).
            outlvl: sets output level of initialization routine
            optarg: solver options dictionary object (default=None, use
                default solver options)
            solver: str indicating which solver to use during initialization
                (default = None)
            hold_state: flag indicating whether the initialization routine
                should unfix any state variables fixed during initialization,
                (default = True). **Valid values:**
                **True** - states variables are not unfixed, and a dict of
                returned containing flags for which states were fixed
                during initialization, **False** - state variables are
                unfixed after initialization by calling the release_state
                method.

        Returns:
            If hold_states is True, returns a dict containing flags for which
            states were fixed during initialization else the release state is
            triggered.
        """
        if optarg is None:
            optarg = {}

        # Get inlet state if not provided
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="control_volume")

        # Provide guesses if none
        if state_args is None:
            blk.estimate_states(always_estimate=True)


        if state_args is None:
            # If no initial guesses provided, estimate values for states
            blk.estimate_states(always_estimate=True)

        # Initialize state blocks
        flags = blk.properties.initialize(
            state_args=state_args,
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            hold_state=True,
        )

        try:
            # TODO: setting state_vars_fixed may not work for heterogeneous
            # systems where a second control volume is involved, as we cannot
            # assume those state vars are also fixed. For now, heterogeneous
            # reactions should ignore the state_vars_fixed argument and always
            # check their state_vars.
            blk.reactions.initialize(
                outlvl=outlvl,
                optarg=optarg,
                solver=solver,
                state_vars_fixed=True,
            )
        except AttributeError:
            pass

        init_log.info("Initialization Complete")

        # Unfix state variables except for source block
        blk.properties.release_state(flags)

        return {}

def release_state(blk, flags, outlvl=idaeslog.NOTSET):
    # No-op: nothing was fixed at the CV level in our custom initialize
    return