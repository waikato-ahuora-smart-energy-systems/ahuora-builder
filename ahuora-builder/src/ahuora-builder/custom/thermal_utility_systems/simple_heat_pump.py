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
Heat Exchanger Models.
"""

__author__ = "Team Ahuora"



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

_log = idaeslog.getLogger(__name__)


class SimpleHeatPumpInitializer(SingleControlVolumeUnitInitializer):
    """
    Initializer for 0D Heat Exchanger units.

    """

    def initialization_routine(
        self,
        model: Block,
        plugin_initializer_args: dict = None,
        copy_inlet_state: bool = False,
        duty=1000 * pyunits.W,
    ):
        """
        Common initialization routine for 0D Heat Exchangers.

        This routine starts by initializing the hot and cold side properties. Next, the heat
        transfer between the two sides is fixed to an initial guess for the heat duty (provided by the duty
        argument), the associated constraint is deactivated, and the model is then solved. Finally, the heat
        duty is unfixed and the heat transfer constraint reactivated followed by a final solve of the model.

        Args:
            model: Pyomo Block to be initialized
            plugin_initializer_args: dict-of-dicts containing arguments to be passed to plug-in Initializers.
                Keys should be submodel components.
            copy_inlet_state: bool (default=False). Whether to copy inlet state to other states or not
                (0-D control volumes only). Copying will generally be faster, but inlet states may not contain
                all properties required elsewhere.
            duty: initial guess for heat duty to assist with initialization. Can be a Pyomo expression with units.

        Returns:
            Pyomo solver results object
        """
        return super(SingleControlVolumeUnitInitializer, self).initialization_routine(
            model=model,
            plugin_initializer_args=plugin_initializer_args,
            copy_inlet_state=copy_inlet_state,
            duty=duty,
        )

    def initialize_main_model(
        self,
        model: Block,
        copy_inlet_state: bool = False,
        duty=1000 * pyunits.W,
    ):
        """
        Initialization routine for main 0D HX models.

        Args:
            model: Pyomo Block to be initialized.
            copy_inlet_state: bool (default=False). Whether to copy inlet state to other states or not
                (0-D control volumes only). Copying will generally be faster, but inlet states may not contain
                all properties required elsewhere.
            duty: initial guess for heat duty to assist with initialization, default = 1000 W. Can
                be a Pyomo expression with units.

        Returns:
            Pyomo solver results object.

        """
        # Get loggers
        init_log = idaeslog.getInitLogger(
            model.name, self.get_output_level(), tag="unit"
        )
        solve_log = idaeslog.getSolveLogger(
            model.name, self.get_output_level(), tag="unit"
        )

        # Create solver
        solver = self._get_solver()

        self.initialize_control_volume(model.source, copy_inlet_state)
        init_log.info_high("Initialization Step 1a (heat output (sink)) Complete.")

        self.initialize_control_volume(model.sink, copy_inlet_state)
        init_log.info_high("Initialization Step 1b (cold side) Complete.")
        # ---------------------------------------------------------------------
        # Solve unit without heat transfer equation
        model.heat_transfer_equation.deactivate()

        # Check to see if heat duty is fixed
        # We will assume that if the first point is fixed, it is fixed at all points
        if not model.sink.heat[model.flowsheet().time.first()].fixed:
            cs_fixed = False

            model.sink.heat.fix(duty)
            for i in model.source.heat:
                model.source.heat[i].set_value(-duty)
        else:
            cs_fixed = True
            for i in model.source.heat:
                model.source.heat[i].set_value(model.sink.heat[i])

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(model, tee=slc.tee)
        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))
        if not cs_fixed:
            model.sink.heat.unfix()
        model.heat_transfer_equation.activate()
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(model, tee=slc.tee)
        init_log.info("Initialization Completed, {}".format(idaeslog.condition(res)))

        return res


def _make_heat_pump_config(config):
    """
    Declare configuration options for HeatExchangerData block.
    """
    config.declare(
        "source",
        ConfigBlock(
            description="Config block for heat output (sink)",
            doc="""A config block used to construct the heat output (sink) control volume.
This config can be given by the heat output (sink) name instead of source.""",
        ),
    )
    config.declare(
        "sink",
        ConfigBlock(
            description="Config block for cold side",
            doc="""A config block used to construct the cold side control volume.
This config can be given by the cold side name instead of sink.""",
        ),
    )
    _make_heater_config_block(config.source)
    _make_heater_config_block(config.sink)


@declare_process_block_class("SimpleHeatPump", doc="Simple 0D heat pump model.")
class SimpleHeatPumpData(UnitModelBlockData):
    """
    Simple 0D heat pump unit.
    Unit model to transfer heat from one material to another.
    """

    default_initializer = SimpleHeatPumpInitializer

    CONFIG = UnitModelBlockData.CONFIG(implicit=True)
    _make_heat_pump_config(CONFIG)

    def build(self):
        """
        Building model

        Args:
            None
        Returns:
            None
        """
        ########################################################################
        #  Call UnitModel.build to setup dynamics and configure                #
        ########################################################################
        super().build()
        config = self.config

        ########################################################################
        # Add control volumes                                                  #
        ########################################################################
        source = _make_heater_control_volume(
            self,
            "source",
            config.source,
            dynamic=config.dynamic,
            has_holdup=config.has_holdup,
        )
        sink = _make_heater_control_volume(
            self,
            "sink",
            config.sink,
            dynamic=config.dynamic,
            has_holdup=config.has_holdup,
        )

        ########################################################################
        # Add variables                                                        #
        ########################################################################
        # Use heat output (sink) units as basis
        s1_metadata = self.source.config.property_package.get_metadata()

        q_units = s1_metadata.get_derived_units("power")
        temp_units = s1_metadata.get_derived_units("temperature")

        self.work_mechanical = Var(
            self.flowsheet().time,
            domain=PositiveReals,
            initialize=100.0,
            doc="Mechanical work input",
            units=q_units,
        )
        self.coefficient_of_performance = Var(
            domain=PositiveReals,
            initialize=2.0,
            doc="Coefficient of performance",
            units=pyunits.dimensionless,
        )
        self.efficiency = Var(
            domain=PositiveReals,
            initialize=0.5,
            doc="Efficiency (Carnot = 1)",
            units=pyunits.dimensionless,
        )
        self.delta_temperature_lift = Var(
            self.flowsheet().time,
            domain=PositiveReals,
            initialize=10.0,
            doc="Temperature lift between source inlet and sink outlet",
            units=temp_units,
        )
        
        self.approach_temperature = Var(
            domain=PositiveReals,
            initialize=10.0,
            doc="Approach temperature of refrigerant",
            units=temp_units,
        )

        self.heat_duty = Reference(sink.heat)
        ########################################################################
        # Add ports                                                            #
        ########################################################################
        self.add_inlet_port(name="source_inlet", block=source, doc="heat output (sink) inlet")
        self.add_inlet_port(
            name="sink_inlet",
            block=sink,
            doc="Cold side inlet",
        )
        self.add_outlet_port(
            name="source_outlet", block=source, doc="heat output (sink) outlet"
        )
        self.add_outlet_port(
            name="sink_outlet",
            block=sink,
            doc="Cold side outlet",
        )

        ########################################################################
        # Add temperature lift constraints                                     #
        ########################################################################

        @self.Constraint(self.flowsheet().time)
        def delta_temperature_lift_equation(b, t):
            # Refrigerant saturation levels (K)
            T_cond = b.sink.properties_out[0].temperature + b.approach_temperature
            T_evap = b.source.properties_out[0].temperature - b.approach_temperature
            # Positive temperature lift
            return b.delta_temperature_lift[t] == T_cond - T_evap

        ########################################################################
        # Add a unit level energy balance                                      #
        ########################################################################
        @self.Constraint(self.flowsheet().time)
        def unit_heat_balance(b, t):
            # The duty of the sink = the source + work input
            return 0 == (
                source.heat[t] + pyunits.convert(sink.heat[t], to_units=q_units) - b.work_mechanical[t]
            )

        
        ########################################################################
        # Add Heat transfer equation                                           #
        ########################################################################

        @self.Constraint(self.flowsheet().time)
        def heat_transfer_equation(b, t):
            # This equation defines the heat duty using the cop of heating.
            return sink.heat[t] == b.coefficient_of_performance * b.work_mechanical[t]

        ########################################################################
        # Add COP Equation                                                     #
        ########################################################################
        @self.Constraint()
        def cop_equation(b):
            # Refrigerant saturation levels (K)
            T_cond = b.sink.properties_out[0].temperature + b.approach_temperature
            T_evap = b.source.properties_out[0].temperature - b.approach_temperature
            # Carnot CoP with efficiency
            return b.coefficient_of_performance == (T_cond / (T_cond - T_evap)) * b.efficiency
    
        
        ########################################################################
        # Add symbols for LaTeX equation rendering                             #
        ########################################################################
        self.work_mechanical.latex_symbol = "W_{mech}"
        self.coefficient_of_performance.latex_symbol = "COP"
        self.efficiency.latex_symbol = "\\eta"
        self.heat_duty.latex_symbol = "Q_{HP}"
        self.delta_temperature_lift.latex_symbol = "\\Delta T_{lift}"
     

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
                initialization for the heat output (sink) (see documentation of the specific
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
        # Set solver options
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        flags1 = self.source.initialize(
            outlvl=outlvl, optarg=optarg, solver=solver, state_args=state_args_1
        )

        init_log.info_high("Initialization Step 1a (heat output (sink)) Complete.")

        flags2 = self.sink.initialize(
            outlvl=outlvl, optarg=optarg, solver=solver, state_args=state_args_2
        )

        init_log.info_high("Initialization Step 1b (cold side) Complete.")
        # ---------------------------------------------------------------------
        # Solve unit without heat transfer equation
        self.heat_transfer_equation.deactivate()

        # Get side 1 and side 2 heat units, and convert duty as needed
        s1_units = self.source.heat.get_units()
        s2_units = self.sink.heat.get_units()

        # Check to see if heat duty is fixed
        # WE will assume that if the first point is fixed, it is fixed at all points
        if not self.sink.heat[self.flowsheet().time.first()].fixed:
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

            self.sink.heat.fix(s2_duty)
            for i in self.source.heat:
                self.source.heat[i].value = s1_duty
        else:
            cs_fixed = True
            for i in self.source.heat:
                self.source.heat[i].set_value(self.sink.heat[i])

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))
        if not cs_fixed:
            self.sink.heat.unfix()
        self.heat_transfer_equation.activate()
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        # ---------------------------------------------------------------------
        # Release Inlet state
        self.source.release_state(flags1, outlvl=outlvl)
        self.sink.release_state(flags2, outlvl=outlvl)

        init_log.info("Initialization Completed, {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )

    def _get_performance_contents(self, time_point=0):
        # this shows as the performance table when calling the report method
        var_dict = {
            
            "Coefficient of Performance": self.coefficient_of_performance,
            "Efficiency": self.efficiency,
            "Source Heat Duty": self.source.heat[time_point],
            "Mechanical Work Input": self.work_mechanical[time_point],
            "Sink Heat Duty": self.sink.heat[time_point],
            "Temperature Lift": self.delta_temperature_lift[time_point],
        }

        expr_dict = {}

        return {"vars": var_dict, "exprs": expr_dict}

    def _get_stream_table_contents(self, time_point=0):
        # this shows as t he stream table when calling the report method
        # Get names for hot and cold sides
        return create_stream_table_dataframe(
            {
                f"Source Inlet": self.source_inlet,
                f"Source Outlet": self.source_outlet,
                f"Sink Inlet": self.sink_inlet,
                f"Sink Outlet": self.sink_outlet,
            },
            time_point=time_point,
        )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        # TODO: Review this code to check it makes sense.

        # Scaling for heat pump variables
        # Mechanical work input: typical values vary, set default scaling to 0.01
        sf_work = dict(
            zip(
            self.work_mechanical.keys(),
            [
                iscale.get_scaling_factor(v, default=0.01)
                for v in self.work_mechanical.values()
            ],
            )
        )
        # COP and efficiency are dimensionless, usually between 1 and 10, default scaling 0.1
        sf_cop = iscale.get_scaling_factor(self.coefficient_of_performance, default=0.1)
        sf_eff = iscale.get_scaling_factor(self.efficiency, default=0.1)

        # Delta Ts: typical range 1-100, default scaling 0.1
        sf_dT_in = dict(
            zip(
            self.delta_temperature_in.keys(),
            [
                iscale.get_scaling_factor(v, default=0.1)
                for v in self.delta_temperature_in.values()
            ],
            )
        )
        sf_dT_out = dict(
            zip(
            self.delta_temperature_out.keys(),
            [
                iscale.get_scaling_factor(v, default=0.1)
                for v in self.delta_temperature_out.values()
            ],
            )
        )

        # Heat duty: depends on process, default scaling 0.01
        sf_q = dict(
            zip(
            self.heat_duty.keys(),
            [
                iscale.get_scaling_factor(v, default=0.01)
                for v in self.heat_duty.values()
            ],
            )
        )

        # Apply scaling to constraints
        for t, c in self.heat_transfer_equation.items():
            iscale.constraint_scaling_transform(
            c, sf_cop * sf_work[t], overwrite=False
            )

        for t, c in self.unit_heat_balance.items():
            iscale.constraint_scaling_transform(
            c, sf_q[t], overwrite=False
            )

        for t, c in self.delta_temperature_in_equation.items():
            iscale.constraint_scaling_transform(c, sf_dT_in[t], overwrite=False)

        for t, c in self.delta_temperature_out_equation.items():
            iscale.constraint_scaling_transform(c, sf_dT_out[t], overwrite=False)

        # COP equation scaling
        iscale.constraint_scaling_transform(
            self.cop_equation, sf_cop, overwrite=False
        )