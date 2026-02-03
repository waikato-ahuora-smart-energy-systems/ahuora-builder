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
Ben Lincolns heat exchanger model using effectiveness

"""

# Import Pyomo libraries
from pyomo.environ import (
    Block,
    check_optimal_termination,
    Constraint,
    Expression,
    Param,
    PositiveReals,
    Reference,
    units as pyunits,
    Var,
)
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.models.unit_models.heat_exchanger import hx_process_config, add_hx_references
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.math import smooth_min, smooth_max
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import InitializationError
import idaes.logger as idaeslog
from idaes.core.initialization import SingleControlVolumeUnitInitializer
from idaes.core.util.model_statistics import degrees_of_freedom
from .inverted import add_inverted, initialise_inverted

__author__ = "Paul Akula, Andrew Lee, Ben Lincoln"


# Set up logger
_log = idaeslog.getLogger(__name__)


class HXEFFInitializer(SingleControlVolumeUnitInitializer):
    """
    Initializer for NTU Heat Exchanger units.

    """

    def initialization_routine(
        self,
        model: Block,
        plugin_initializer_args: dict = None,
        copy_inlet_state: bool = False,
        duty=1000 * pyunits.W,
    ):
        """
        Common initialization routine for NTU Heat Exchangers.

        This routine starts by initializing the hot and cold side properties. Next, the heat
        transfer between the two sides is fixed to an initial guess for the heat duty (provided by the duty
        argument), the associated constraint deactivated, and the model is then solved. Finally, the heat
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
        Initialization routine for main NTU HX models.

        Args:
            model: Pyomo Block to be initialized.
            copy_inlet_state: bool (default=False). Whether to copy inlet state to other states or not
                (0-D control volumes only). Copying will generally be faster, but inlet states may not contain
                all properties required elsewhere.
            duty: initial guess for heat duty to assist with initialization. Can be a Pyomo expression with units.

        Returns:
            Pyomo solver results object.

        """
        initialise_inverted(model.hot_side,"deltaP")
        initialise_inverted(model.cold_side,"deltaP")
        # TODO: Aside from one differences in constraint names, this is
        # identical to the Initializer for the 0D HX unit.
        # Set solver options
        init_log = idaeslog.getInitLogger(
            model.name, self.get_output_level(), tag="unit"
        )
        solve_log = idaeslog.getSolveLogger(
            model.name, self.get_output_level(), tag="unit"
        )

        # Create solver
        solver = self._get_solver()

        self.initialize_control_volume(model.hot_side, copy_inlet_state)
        init_log.info_high("Initialization Step 1a (hot side) Complete.")

        self.initialize_control_volume(model.cold_side, copy_inlet_state)
        init_log.info_high("Initialization Step 1b (cold side) Complete.")

        # ---------------------------------------------------------------------
        # Solve unit without heat transfer equation
        model.energy_balance_constraint.deactivate()

        model.cold_side.heat.fix(duty)
        for i in model.hot_side.heat:
            model.hot_side.heat[i].set_value(-duty)

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(model, tee=slc.tee)

        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        model.cold_side.heat.unfix()
        model.energy_balance_constraint.activate()
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(model, tee=slc.tee)
        init_log.info("Initialization Completed, {}".format(idaeslog.condition(res)))

        return res


@declare_process_block_class("HeatExchangerEffectiveness")
class HeatExchangerEFFData(UnitModelBlockData):
    """Heat Exchanger Unit Model using NTU method."""

    default_initializer = HXEFFInitializer

    CONFIG = UnitModelBlockData.CONFIG(implicit=True)

    # Configuration template for fluid specific  arguments
    _SideCONFIG = ConfigBlock()
    _SideCONFIG.declare(
        "has_phase_equilibrium",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Phase equilibrium construction flag",
            doc="""Indicates whether terms for phase equilibrium should be
constructed, **default** = False.
**Valid values:** {
**True** - include phase equilibrium terms
**False** - exclude phase equilibrium terms.}""",
        ),
    )
    _SideCONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.useDefault.
**Valid values:** {
**MaterialBalanceType.useDefault - refer to property package for default
balance type
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    _SideCONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.useDefault,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
**default** - EnergyBalanceType.useDefault.
**Valid values:** {
**EnergyBalanceType.useDefault - refer to property package for default
balance type
**EnergyBalanceType.none** - exclude energy balances,
**EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
**EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
**EnergyBalanceType.energyTotal** - single energy balance for material,
**EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )
    _SideCONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )
    _SideCONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}""",
        ),
    )
    _SideCONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use ",
            doc="""Property parameter object used to define property calculations
        **default** - useDefault.
        **Valid values:** {
        **useDefault** - use default package from parent model or flowsheet,
        **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    _SideCONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property package",
            doc="""A ConfigBlock with arguments to be passed to
        property block(s) and used when constructing these,
        **default** - None.
        **Valid values:** {
        see property package for documentation.}""",
        ),
    )

    # Create individual config blocks for hot and cold sides
    CONFIG.declare("hot_side", _SideCONFIG(doc="Hot fluid config arguments"))
    CONFIG.declare("cold_side", _SideCONFIG(doc="Cold fluid config arguments"))
    CONFIG.declare(
        "hot_side_name",
        ConfigValue(
            default=None,
            domain=str,
            doc="Hot side name, sets control volume and inlet and outlet names",
        ),
    )
    CONFIG.declare(
        "cold_side_name",
        ConfigValue(
            default=None,
            domain=str,
            doc="Cold side name, sets control volume and inlet and outlet names",
        ),
    )

    def build(self):
        # Call UnitModel.build to setup model
        super().build()
        hx_process_config(self)

        # ---------------------------------------------------------------------
        # Build hot-side control volume
        self.hot_side = ControlVolume0DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.hot_side.property_package,
            property_package_args=self.config.hot_side.property_package_args,
        )

        # TODO : Add support for phase equilibrium?
        self.hot_side.add_state_blocks(has_phase_equilibrium=self.config.hot_side.has_phase_equilibrium)

        self.hot_side.add_material_balances(
            balance_type=self.config.hot_side.material_balance_type,
            has_phase_equilibrium=self.config.hot_side.has_phase_equilibrium,
        )

        self.hot_side.add_energy_balances(
            balance_type=self.config.hot_side.energy_balance_type,
            has_heat_transfer=True,
        )

        self.hot_side.add_momentum_balances(
            balance_type=self.config.hot_side.momentum_balance_type,
            has_pressure_change=self.config.hot_side.has_pressure_change,
        )

        # ---------------------------------------------------------------------
        # Build cold-side control volume
        self.cold_side = ControlVolume0DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.cold_side.property_package,
            property_package_args=self.config.cold_side.property_package_args,
        )

        self.cold_side.add_state_blocks(has_phase_equilibrium=self.config.cold_side.has_phase_equilibrium)

        self.cold_side.add_material_balances(
            balance_type=self.config.cold_side.material_balance_type,
            has_phase_equilibrium=self.config.cold_side.has_phase_equilibrium,
        )

        self.cold_side.add_energy_balances(
            balance_type=self.config.cold_side.energy_balance_type,
            has_heat_transfer=True,
        )

        self.cold_side.add_momentum_balances(
            balance_type=self.config.cold_side.momentum_balance_type,
            has_pressure_change=self.config.cold_side.has_pressure_change,
        )

        # ---------------------------------------------------------------------
        # Add Ports to control volumes
        self.add_inlet_port(
            name="hot_side_inlet", block=self.hot_side, doc="Hot side inlet port"
        )
        self.add_outlet_port(
            name="hot_side_outlet", block=self.hot_side, doc="Hot side outlet port"
        )

        self.add_inlet_port(
            name="cold_side_inlet", block=self.cold_side, doc="Cold side inlet port"
        )
        self.add_outlet_port(
            name="cold_side_outlet", block=self.cold_side, doc="Cold side outlet port"
        )

        # ---------------------------------------------------------------------
        # Add unit level References
        # Set references to balance terms at unit level
        self.heat_duty = Reference(self.cold_side.heat[:])

        # Add references to the user provided aliases (if applicable).
        add_hx_references(self)

        # ---------------------------------------------------------------------
        # Add performance equations
        # All units of measurement will be based on hot side
        hunits = self.hot_side.config.property_package.get_metadata().get_derived_units

        # Overall energy balance
        def rule_energy_balance(blk, t):
            return blk.hot_side.heat[t] == -pyunits.convert(
                blk.cold_side.heat[t], to_units=hunits("power")
            )

        self.energy_balance_constraint = Constraint(
            self.flowsheet().time, rule=rule_energy_balance
        )

        # Add e-NTU variables
        self.effectiveness = Var(
            self.flowsheet().time,
            initialize=1,
            units=pyunits.dimensionless,
            domain=PositiveReals,
            doc="Effectiveness factor",
        )

        # Minimum heat capacitance ratio for e-NTU method
        self.eps_cmin = Param(
            initialize=1e-3,
            mutable=True,
            units=hunits("power") / hunits("temperature"),
            doc="Epsilon parameter for smooth Cmin and Cmax",
        )
        
###########

        # Todo: Will not work for a pure component system in the two phase region

###########

        #Build hotside State Block
        tmp_dict = dict(**self.config.hot_side.property_package_args)
        tmp_dict["has_phase_equilibrium"] = self.config.hot_side.has_phase_equilibrium
        tmp_dict["defined_state"] = False

        self.properties_hotside = self.config.hot_side.property_package.build_state_block(
            self.flowsheet().time, doc="Hot side properties at outlet under Qmax", **tmp_dict
        )
        #Connect Properties to Hot Side 
        @self.Constraint(
                self.flowsheet().time, doc="Hot Side Temperature at Qmax = Inlet Cold Temp")
        def constraint_hotside_temp(self, t):
            return self.properties_hotside[t].temperature == self.cold_side.properties_in[t].temperature
        @self.Constraint(
                self.flowsheet().time, doc="Hot Side Pressure at Qmax = Inlet Hot Pressure")
        def constraint_hotside_pres(self, t):
            return self.properties_hotside[t].pressure == self.hot_side.properties_in[t].pressure
        @self.Constraint(
                self.flowsheet().time, doc="Hot Side Flowrate at Qmax = Inlet Hot Flowrate")
        def constraint_hotside_flow(self, t):
            return self.properties_hotside[t].flow_mol == self.hot_side.properties_in[t].flow_mol
        
        #Build Coldside State Block
        tmp_dict = dict(**self.config.cold_side.property_package_args)
        tmp_dict["has_phase_equilibrium"] = self.config.cold_side.has_phase_equilibrium
        tmp_dict["defined_state"] = False

        self.properties_coldside = self.config.cold_side.property_package.build_state_block(
            self.flowsheet().time, doc="Cold side properties at outlet under Qmax", **tmp_dict
        )

########

        # Todo: Will not work for a pure component system in the two phase region

###########

        #Connect Properties to Cold Side
        @self.Constraint(
                self.flowsheet().time, doc="Cold Side Temperature at Qmax = Inlet Hot Temp")
        def constraint_coldside_temp(self, t):
            return self.properties_coldside[t].temperature == self.hot_side.properties_in[t].temperature
        @self.Constraint(
                self.flowsheet().time, doc="Cold Side Pressure at Qmax = Inlet Cold Pressure")
        def constraint_coldside_pres(self, t):
            return self.properties_coldside[t].pressure == self.cold_side.properties_in[t].pressure
        @self.Constraint(
                self.flowsheet().time, doc="Cold Side Flowrate at Qmax = Inlet Cold Flowrate")
        def constraint_coldside_flow(self, t):
            return self.properties_coldside[t].flow_mol == self.cold_side.properties_in[t].flow_mol
        
        # Delta h hot at Qmax
        def rule_deltah_hot_qmax(blk, t):
            return blk.hot_side.properties_in[t].enth_mol - blk.properties_hotside[t].enth_mol

        self.delta_h_hot_qmax = Expression(
            self.flowsheet().time, rule=rule_deltah_hot_qmax, doc="Delta h Hot Side at Qmax"
        )
        # Delta h cold at Qmax
        def rule_deltah_cold_qmax(blk, t):
            return blk.properties_coldside[t].enth_mol - blk.cold_side.properties_in[t].enth_mol
        self.delta_h_cold_qmax = Expression(
            self.flowsheet().time, rule=rule_deltah_cold_qmax, doc="Delta h Cold Side at Qmax"
        )

        # TODO : Support both mass and mole based flows
        # Minimum heat transfer rate
        def rule_Hmin(blk, t):
            Hhot = pyunits.convert(
                blk.hot_side.properties_in[t].flow_mol
                * blk.delta_h_hot_qmax[t],
                to_units=hunits("power"),
            )
            Hcold = pyunits.convert(
                blk.cold_side.properties_in[t].flow_mol
                * blk.delta_h_cold_qmax[t],
                to_units=hunits("power"),
            )
            return smooth_min(Hhot, Hcold, eps=blk.eps_cmin)

        self.Qmax = Expression(
            self.flowsheet().time, rule=rule_Hmin, doc="Max heat transfer rate"
        )
        #Max heat transfer rate
        def rule_Hmax(blk, t):
            Hhot = pyunits.convert(
                blk.hot_side.properties_in[t].flow_mol
                * blk.delta_h_hot_qmax[t],
                to_units=hunits("power"),
            )
            Hcold = pyunits.convert(
                blk.cold_side.properties_in[t].flow_mol
                * blk.delta_h_cold_qmax[t],
                to_units=hunits("power"),
            )
            return smooth_max(Hhot, Hcold, eps=blk.eps_cmin)
        self.Hmax_theory = Expression(
            self.flowsheet().time,
              rule=rule_Hmax, doc="Maximum heat transfer rate"
        )

# Add effectiveness relation
        def rule_effectiveness(blk, t):
            return blk.hot_side.heat[t] == -(
                blk.effectiveness[t]
                * blk.Qmax[t]
            )

        self.heat_duty_constraint = Constraint(self.flowsheet().time, rule=rule_effectiveness)

        add_inverted(self.hot_side,"deltaP")
        add_inverted(self.cold_side,"deltaP")
    # TODO : Add scaling methods

    def initialize_build(
        self,
        hot_side_state_args=None,
        cold_side_state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
        duty=None,
    ):
        """
        Heat exchanger initialization method.

        Args:
            hot_side_state_args : a dict of arguments to be passed to the
                property initialization for the hot side (see documentation of
                the specific property package) (default = None).
            cold_side_state_args : a dict of arguments to be passed to the
                property initialization for the cold side (see documentation of
                the specific property package) (default = None).
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
        initialise_inverted(self.hot_side,"deltaP")
        initialise_inverted(self.cold_side,"deltaP")
        # Set solver options
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        hot_side = self.hot_side
        cold_side = self.cold_side
        hot_side_max = self.properties_hotside
        cold_side_max = self.properties_coldside
        

        # Create solver
        opt = get_solver(solver, optarg)

        flags1 = hot_side.initialize(
            outlvl=outlvl, optarg=optarg, solver=solver, state_args=hot_side_state_args
        )

        init_log.info_high("Initialization Step 1a (hot side) Complete.")

        flags2 = cold_side.initialize(
            outlvl=outlvl, optarg=optarg, solver=solver, state_args=cold_side_state_args
        )

        init_log.info_high("Initialization Step 1b (cold side) Complete.")

        # cold_side_max[0].pressure.set_value(cold_side.properties_in[0].pressure.value)
        # hot_side_max[0].pressure.set_value(hot_side.properties_in[0].pressure.value)

        # cold_side_max[0].flow_mol.set_value(cold_side.properties_in[0].flow_mol.value)
        # hot_side_max[0].flow_mol.set_value(hot_side.properties_in[0].flow_mol.value)

        # cold_side_max[0].enth_mol.set_value(cold_side.properties_in[0].enth_mol.value)
        # hot_side_max[0].enth_mol.set_value(hot_side.properties_in[0].enth_mol.value)

        flags3 = hot_side_max.initialize(
            outlvl=outlvl
        )
        init_log.info_high("Initialization Step 1c (hot side max) Complete.")

        flags4 = cold_side_max.initialize(
            outlvl=outlvl
        )
        init_log.info_high("Initialization Step 1d (cold side max) Complete.")

        hot_side_max[0].display()
        cold_side_max[0].display()


        #----------------------------------------------------------------------
       
        self.energy_balance_constraint.deactivate()
        self.heat_duty_constraint.deactivate()
        self.hot_side.enthalpy_balances.deactivate()
        self.cold_side.enthalpy_balances.deactivate()
        self.cold_side.material_balances.deactivate()
        self.hot_side.material_balances.deactivate()
        self.hot_side.properties_out.deactivate()
        self.cold_side.properties_out.deactivate()

        print("DOF:", degrees_of_freedom(self))
        from idaes.core.util import DiagnosticsToolbox
        dt = DiagnosticsToolbox(self)
        dt.display_underconstrained_set()
        dt.display_overconstrained_set()
        dt.report_structural_issues()
        dt.report_numerical_issues()
        dt.display_constraints_with_large_residuals()
        #dt.compute_infeasibility_explanation()
        dt.display_variables_at_or_outside_bounds()

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
            

        init_log.info_high("Initialization Step 1.5 {}.".format(idaeslog.condition(res)))

        # ---------------------------------------------------------------------
        # Solve unit without heat transfer equation
        self.heat_duty_constraint.activate()
        self.heat_duty.unfix()
        self.hot_side.enthalpy_balances.activate()
        self.cold_side.enthalpy_balances.activate()
        self.cold_side.material_balances.activate()
        self.hot_side.material_balances.activate()
        self.hot_side.properties_out.activate()
        self.cold_side.properties_out.activate()
        
        # self.Qmax[0].set_value(1000)

        # Get side 1 and side 2 heat units, and convert duty as needed
        s1_units = hot_side.heat.get_units()
        s2_units = cold_side.heat.get_units()

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

        cold_side.heat.fix(s2_duty)
        for i in hot_side.heat:
            hot_side.heat[i].value = s1_duty

        print("DOF:", degrees_of_freedom(self))

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)

        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        cold_side.heat.unfix()
        self.energy_balance_constraint.activate()

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))

        # ---------------------------------------------------------------------
        # Release Inlet state
        hot_side.release_state(flags1, outlvl=outlvl)
        cold_side.release_state(flags2, outlvl=outlvl)

        init_log.info("Initialization Completed, {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )

        return res

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Hot Inlet": self.hot_side_inlet,
                "Hot Outlet": self.hot_side_outlet,
                "Cold Inlet": self.cold_side_inlet,
                "Cold Outlet": self.cold_side_outlet,
            },
            time_point=time_point,
        )