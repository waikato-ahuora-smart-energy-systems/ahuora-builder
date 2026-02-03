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
Standard IDAES pressure changer model.
"""
# TODO: Missing docstrings
# pylint: disable=missing-function-docstring

# Changing existing config block attributes
# pylint: disable=protected-access

# Import Python libraries


# Import Pyomo libraries
from pyomo.environ import (
    Block,
    value,
    Var,
    Expression,
    Constraint,
    Reference,
    check_optimal_termination,
    Reals,
    Param,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    EnergyBalanceType,
    MomentumBalanceType,
    MaterialBalanceType,
    ProcessBlockData,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.exceptions import PropertyNotSupportedError, InitializationError
from idaes.core.util.config import is_physical_parameter_block
import idaes.logger as idaeslog
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
from idaes.core.initialization import SingleControlVolumeUnitInitializer
from idaes.core.util import to_json, from_json, StoreSpec
from idaes.core.util.math import smooth_max, safe_sqrt, sqrt, smooth_min
from pyomo.environ import units as pyunits


__author__ = "Emmanuel Ogbe, Andrew Lee"
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("TurbineBase")
class TurbineBaseData(UnitModelBlockData):
    """
    Standard Compressor/Expander Unit Model Class
    """

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
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
    CONFIG.declare(
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
    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be
constructed, **default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )
    CONFIG.declare(
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
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property
calculations, **default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property
block(s) and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "calculation_method",
        ConfigValue(
            default='isentropic',
            domain=In(["isentropic", "simple_willans", "part_load_willans", "Tsat_willans", "BPST_willans", "CT_willans"]),
            description="Calculation method used to model mechanical work",
            doc="""Property parameter object used to define property
calculations, **default** - 'isentropic'.
**Valid values:** {
**isentropic** - default method, uses isentropic efficiency to determine work
**simple_willans** - simple willans line requring slope, intercept and max molar flow.
**part_load_willans** - willans line with part load correction, a, b, c and max molar flow as parameters
**Tsat_willans** - willans line with part load correction using saturation temperature. Requires only max molar flow
**BPST_willans** - back pressure willans line with part load correction using pressure difference, requires max molar flow.
**CT_willans** - condensing turbine willans line with part load correction using pressure difference, requires max molar flow.
}""",
        ),
    )

    def build(self):
        """

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build
        super().build()

        # Add a control volume to the unit including setting up dynamics.
        self.control_volume = ControlVolume0DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )

        # Add geometry variables to control volume
        if self.config.has_holdup:
            self.control_volume.add_geometry()

        # Add inlet and outlet state blocks to control volume
        self.control_volume.add_state_blocks(
            has_phase_equilibrium=self.config.has_phase_equilibrium
        )

        # Add mass balance
        # Set has_equilibrium is False for now
        # TO DO; set has_equilibrium to True
        self.control_volume.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_phase_equilibrium=self.config.has_phase_equilibrium,
        )

        # Add energy balance
        eb = self.control_volume.add_energy_balances(
            balance_type=self.config.energy_balance_type, has_work_transfer=True
        )

        # add momentum balance
        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type, has_pressure_change=True
        )

        # Add Ports
        self.add_inlet_port()
        self.add_outlet_port()

        # Set Unit Geometry and holdup Volume
        if self.config.has_holdup is True:
            self.volume = Reference(self.control_volume.volume[:])

        self.work_mechanical = Reference(self.control_volume.work[:])
        # self.work_mechanical.fix(-1000e3)
        # self.work_mechanical.unfix()


        # Add Momentum balance variable 'deltaP'
        self.deltaP = Reference(self.control_volume.deltaP[:])

        # Performance Variables
        self.ratioP = Var(self.flowsheet().time, initialize=1.0, doc="Pressure Ratio")

        # Pressure Ratio
        @self.Constraint(self.flowsheet().time, doc="Pressure ratio constraint")
        def ratioP_calculation(self, t):
            return (
                self.ratioP[t] * self.control_volume.properties_in[t].pressure
                == self.control_volume.properties_out[t].pressure
            )

        units_meta = self.control_volume.config.property_package.get_metadata()

        # Get indexing sets from control volume
        # Add isentropic variables
        self.efficiency_isentropic = Var(
            self.flowsheet().time,
            initialize=0.5,
            doc="Efficiency with respect to an isentropic process [-]",
        )

        # self.delta_h_is = Var(
        #     self.flowsheet().time,
        #     initialize=-100e3,
        #     doc="Enthalpy difference input to unit if isentropic process",
        #     units=units_meta.get_derived_units("energy") / units_meta.get_derived_units("amount"),
        # )
        # self.delta_h_act = Var(
        #     self.flowsheet().time,
        #     initialize=-100e3,
        #     doc="Enthalpy difference input to unit for actual process",
        #     units=units_meta.get_derived_units("energy") / units_meta.get_derived_units("amount"),
        # )
        # self.work_isentropic = Var(
        #     self.flowsheet().time,
        #     initialize=-100e3,
        #     doc="Work input to unit if isentropic process",
        #     units=units_meta.get_derived_units("power"),
        # )

        # Add motor/electrical work and efficiency variable
        self.efficiency_motor = Var(
            self.flowsheet().time,
            initialize=1.0,
            doc="Motor efficiency converting shaft work to electrical work [-]",
            )
        
        self.work_electrical = Var(
            self.flowsheet().time,
            initialize=1.0,
            doc="Electrical work of a turbine [-]",
            units=units_meta.get_derived_units("power")
            )

        # Add willans line parameters
        if 'willans' in self.config.calculation_method:
            self.willans_slope = Var(
                self.flowsheet().time,
                initialize=100,
                doc="Slope of willans line",
                units=units_meta.get_derived_units("energy") / units_meta.get_derived_units("amount"),
            )

            self.willans_intercept = Var(
                self.flowsheet().time,
                initialize=-100,
                doc="Intercept of willans line",
                units=units_meta.get_derived_units("power"),
            )

            self.willans_max_mol = Var(
                self.flowsheet().time,
                initialize=1.0,
                doc="Max molar flow of willans line",
                units=units_meta.get_derived_units("amount") / units_meta.get_derived_units("time"),
            )

            if self.config.calculation_method in ["part_load_willans", "Tsat_willans", "BPST_willans", "CT_willans"]:
                self.willans_a = Var(
                    self.flowsheet().time,
                    initialize=1.0,
                    doc="Willans a coefficient",
                )

                self.willans_b = Var(
                    self.flowsheet().time,
                    initialize=1.0,
                    doc="Willans b coefficient",
                    units=units_meta.get_derived_units("power")
                )

                self.willans_efficiency = Var(
                    self.flowsheet().time,
                    initialize=1.0,
                    doc="Willans efficiency",
                )

        # Build isentropic state block
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = self.config.has_phase_equilibrium
        tmp_dict["defined_state"] = False

        self.properties_isentropic = self.config.property_package.build_state_block(
            self.flowsheet().time, doc="isentropic properties at outlet", **tmp_dict
        )

        # Connect isentropic state block properties
        @self.Constraint(
            self.flowsheet().time, doc="Pressure for isentropic calculations"
        )
        def isentropic_pressure(self, t):
            return (
                self.properties_isentropic[t].pressure
                == self.control_volume.properties_out[t].pressure
            )

        # This assumes isentropic composition is the same as outlet
        self.add_state_material_balances(
            self.config.material_balance_type,
            self.properties_isentropic,
            self.control_volume.properties_out,
        )

        # This assumes isentropic entropy is the same as inlet
        @self.Constraint(self.flowsheet().time, doc="Isentropic assumption")
        def isentropic(self, t):
            return (
                self.properties_isentropic[t].entr_mol
                == self.control_volume.properties_in[t].entr_mol
            )
    
        self.add_isentropic_work_definition()
        
        if 'willans' in self.config.calculation_method: 
            # Write isentropic efficiency eqn
            self.add_willans_line_relationship() 

            if self.config.calculation_method in ["part_load_willans", "Tsat_willans", "BPST_willans", "CT_willans"]:
                if self.config.calculation_method == "Tsat_willans": # use published values and dTsat to calculate willans a,b,c
                    self.calculate_Tsat_willans_parameters()

                elif self.config.calculation_method == "BPST_willans":
                    self.calculate_BPST_willans_parameters()
                
                elif self.config.calculation_method == "CT_willans":
                    self.calculate_CT_willans_parameters()

                # calculate slope and intercept
                self.calculate_willans_coefficients()

        self.add_mechanical_and_isentropic_work_definition()
        self.add_electrical_work_definition()
       
    def calculate_CT_willans_parameters(self):
        # a parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans CT a calculation"
        )
        def willans_CT_a_calculation(self, t):
            return self.willans_a[t] == 1.288464 - 0.0015185 * (self.control_volume.properties_in[t].pressure / 1e5) / pyunits.Pa - 0.33415834 * (self.control_volume.properties_out[t].pressure / 1e5) / pyunits.Pa
        
        # b parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans CT b calculation"
        )
        def willans_CT_b_calculation(self, t):
            return self.willans_b[t] == (-437.7746025 + 29.00736723 * (self.control_volume.properties_in[t].pressure / 1e5) / pyunits.Pa + 10.35902331 * (self.control_volume.properties_out[t].pressure / 1e5) / pyunits.Pa) * 1000 * pyunits.W
        
        # c parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans CT efficiency calculation"
        )
        def willans_CT_efficiency_calculation(self, t):
            return self.willans_efficiency[t] == 1 / ((0.07886297 + 0.000528327 * (self.control_volume.properties_in[t].pressure / 1e5) / pyunits.Pa - 0.703153891 * (self.control_volume.properties_out[t].pressure / 1e5) / pyunits.Pa) + 1)      

    def calculate_BPST_willans_parameters(self):
        # a parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans BPST a calculation"
        )
        def willans_BPST_a_calculation(self, t):
            return self.willans_a[t] == 1.18795366 - 0.00029564 * (self.control_volume.properties_in[t].pressure / 1e5) / pyunits.Pa + 0.004647288 * (self.control_volume.properties_out[t].pressure / 1e5) / pyunits.Pa
        

        # b parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans BPST b calculation"
        )
        def willans_BPST_b_calculation(self, t):
            return self.willans_b[t] == (449.9767142 + 5.670176939 * (self.control_volume.properties_in[t].pressure / 1e5) / pyunits.Pa - 11.5045814 * (self.control_volume.properties_out[t].pressure / 1e5) / pyunits.Pa) * 1000 * pyunits.W
        

        # c parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans BPST c calculation"
        )
        def willans_BPST_efficiency_calculation(self, t):
            return self.willans_efficiency[t] == 1 / ((0.205149333 - 0.000695171 * (self.control_volume.properties_in[t].pressure / 1e5) / pyunits.Pa + 0.002844611 * (self.control_volume.properties_out[t].pressure / 1e5) / pyunits.Pa) + 1)       

    def calculate_Tsat_willans_parameters(self):
        # a parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans Tsat a calculation"
        )
        def willans_Tsat_a_calculation(self, t):
            return self.willans_a[t] == (1.155 + 0.000538 * (self.control_volume.properties_in[t].temperature_sat - self.control_volume.properties_out[t].temperature_sat) / pyunits.K)

        # b parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans Tsat b calculation"
        )
        def willans_Tsat_b_calculation(self, t):
            return self.willans_b[t] == (0 + 4.23 * (self.control_volume.properties_in[t].temperature_sat - self.control_volume.properties_out[t].temperature_sat) / pyunits.K)*1000 * pyunits.W
        
        # c parameter
        @self.Constraint(
                self.flowsheet().time, doc="Willans Tsat efficiency calculation"
        )
        def willans_Tsat_c_calculation(self, t):
            return self.willans_efficiency[t] == 0.83333

    def calculate_willans_coefficients(self):
        # Calculate willans coefficients
            @self.Constraint(
                self.flowsheet().time, doc="Willans slope calculation"
            )
            def willans_slope_calculation(self, t):
                return self.willans_slope[t] == 1 / (self.willans_efficiency[t] * self.willans_a[t]) * ((self.control_volume.properties_in[t].enth_mol - self.properties_isentropic[t].enth_mol) - self.willans_b[t] / self.willans_max_mol[t])
            
            
            @self.Constraint(
                self.flowsheet().time, doc="Willans intercept calculation"
            )
            def willans_intercept_calculation(self, t):
                return self.willans_intercept[t] == ((1 - self.willans_efficiency[t]) / (self.willans_efficiency[t] * self.willans_a[t])) * ((self.control_volume.properties_in[t].enth_mol - self.properties_isentropic[t].enth_mol) * self.willans_max_mol[t] - self.willans_b[t])
    
    def add_mechanical_and_isentropic_work_definition(self):
        
        if 'willans' in self.config.calculation_method: 
            self.efficiency_isentropic = Expression(
                self.flowsheet().time,
                rule=lambda b, t: (
                    b.delta_h_act[t] / b.delta_h_is[t]
                )
            )
        else:
            # Mechanical work
            @self.Constraint(
                self.flowsheet().time, doc="Isentropic and mechanical work relationship"
            )
            def isentropic_and_mechanical_work_eq(self, t):
                    return self.work_mechanical[t] == (
                        self.delta_h_is[t] * self.control_volume.properties_in[t].flow_mol * self.efficiency_isentropic[t]
                    )
            
    def add_willans_line_relationship(self):
        @self.Constraint(
            self.flowsheet().time, doc="Willans line and mechanical work relationship"
        )
        def willans_line_eq(self, t):            
            eps = 0.0001  # smoothing parameter; smaller = closer to exact max, larger = smoother
            return self.work_mechanical[t] == smooth_min(
                -(self.willans_slope[t] * self.control_volume.properties_in[t].flow_mol - self.willans_intercept[t]) / (self.willans_slope[t] * self.willans_max_mol[t]),
                0.0,
                eps
                ) * (self.willans_slope[t] * self.willans_max_mol[t])
         
    def add_electrical_work_definition(self):
        # Electrical work
        @self.Constraint(
            self.flowsheet().time, doc="Calculate electrical work of turbine"
        )
        def electrical_energy_balance(self, t):
            return self.work_electrical[t] == self.work_mechanical[t] * self.efficiency_motor[t]
        
    def add_isentropic_work_definition(self):
        self.delta_h_is = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                b.properties_isentropic[t].enth_mol - b.control_volume.properties_in[t].enth_mol
            )
        )
        self.delta_h_act = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                b.control_volume.properties_out[t].enth_mol - b.control_volume.properties_in[t].enth_mol
            )
        )
        # # Isentropic work
        # @self.Constraint(
        #     self.flowsheet().time, doc="Calculate work of isentropic process"
        # )
        # def isentropic_energy_balance(self, t):
        #     return self.work_isentropic[t] == ( self.properties_isentropic[t].enth_mol - self.control_volume.properties_in[t].enth_mol ) * self.control_volume.properties_in[t].flow_mol
            
    def initialize_build(
        blk,
        state_args=None,
        routine=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for pressure changer initialization routines

        Keyword Arguments:
            routine : str stating which initialization routine to execute
                        * None - use routine matching thermodynamic_assumption
                        * 'isentropic' - use isentropic initialization routine
                        * 'isothermal' - use isothermal initialization routine
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)

        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        cv = blk.control_volume
        t0 = blk.flowsheet().time.first()
        state_args_out = {}

        if state_args is None:
            state_args = {}
            state_dict = cv.properties_in[t0].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        # Get initialisation guesses for outlet and isentropic states
        for k in state_args:
            if k == "pressure" and k not in state_args_out:
                # Work out how to estimate outlet pressure
                if cv.properties_out[t0].pressure.fixed:
                    # Fixed outlet pressure, use this value
                    state_args_out[k] = value(cv.properties_out[t0].pressure)
                elif blk.deltaP[t0].fixed:
                    state_args_out[k] = value(state_args[k] + blk.deltaP[t0])
                elif blk.ratioP[t0].fixed:
                    state_args_out[k] = value(state_args[k] * blk.ratioP[t0])
                else:
                    # Not obvious what to do, use inlet state
                    state_args_out[k] = state_args[k]
            elif k not in state_args_out:
                state_args_out[k] = state_args[k]

        # Initialize state blocks
        flags = cv.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            hold_state=True,
            state_args=state_args,
        )
        cv.properties_out.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            hold_state=False,
            state_args=state_args_out,
        )

        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        # Initialize Isentropic block

        blk.properties_isentropic.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_out,
        )

        init_log.info_high("Initialization Step 2 Complete.")

        # Skipping step 3 because Isothermal had problems.

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high("Initialization Step 4 {}.".format(idaeslog.condition(res)))


        # ---------------------------------------------------------------------
        # Release Inlet state
        blk.control_volume.release_state(flags, outlvl)

        if not check_optimal_termination(res):
            raise InitializationError(
                f"{blk.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )

        init_log.info(f"Initialization Complete: {idaeslog.condition(res)}")

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        if hasattr(self, "deltaP"):
            var_dict["Mechanical Work"] = self.work_mechanical[time_point]
        if hasattr(self, "deltaP"):
            var_dict["Electrical Work"] = self.work_electrical[time_point]
        if hasattr(self, "deltaP"):
            var_dict["Pressure Change"] = self.deltaP[time_point]
        if hasattr(self, "ratioP"):
            var_dict["Pressure Ratio"] = self.ratioP[time_point]
        # if hasattr(self, "efficiency_isentropic"):
        #     var_dict["Isentropic Efficiency"] = self.efficiency_isentropic[time_point]

        return {"vars": var_dict}

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        if hasattr(self, "work_fluid"):
            for t, v in self.work_fluid.items():
                iscale.set_scaling_factor(
                    v,
                    iscale.get_scaling_factor(
                        self.control_volume.work[t], default=1, warning=True
                    ),
                )

        if hasattr(self, "work_mechanical"):
            for t, v in self.work_mechanical.items():
                iscale.set_scaling_factor(
                    v,
                    iscale.get_scaling_factor(
                        self.control_volume.work[t], default=1, warning=True
                    ),
                )

        if hasattr(self, "work_isentropic"):
            for t, v in self.work_isentropic.items():
                iscale.set_scaling_factor(
                    v,
                    iscale.get_scaling_factor(
                        self.control_volume.work[t], default=1, warning=True
                    ),
                )

        if hasattr(self, "ratioP_calculation"):
            for t, c in self.ratioP_calculation.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.properties_in[t].pressure,
                        default=1,
                        warning=True,
                    ),
                    overwrite=False,
                )

        if hasattr(self, "fluid_work_calculation"):
            for t, c in self.fluid_work_calculation.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.deltaP[t], default=1, warning=True
                    ),
                    overwrite=False,
                )

        if hasattr(self, "actual_work"):
            for t, c in self.actual_work.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.work[t], default=1, warning=True
                    ),
                    overwrite=False,
                )

        if hasattr(self, "isentropic_pressure"):
            for t, c in self.isentropic_pressure.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.properties_in[t].pressure,
                        default=1,
                        warning=True,
                    ),
                    overwrite=False,
                )

        if hasattr(self, "isentropic"):
            for t, c in self.isentropic.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.properties_in[t].entr_mol,
                        default=1,
                        warning=True,
                    ),
                    overwrite=False,
                )

        if hasattr(self, "isentropic_energy_balance"):
            for t, c in self.isentropic_energy_balance.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.work[t], default=1, warning=True
                    ),
                    overwrite=False,
                )

        if hasattr(self, "zero_work_equation"):
            for t, c in self.zero_work_equation.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.control_volume.work[t], default=1, warning=True
                    ),
                )

        if hasattr(self, "state_material_balances"):
            cvol = self.control_volume
            phase_list = cvol.properties_in.phase_list
            phase_component_set = cvol.properties_in.phase_component_set
            mb_type = cvol._constructed_material_balance_type
            if mb_type == MaterialBalanceType.componentPhase:
                for (t, p, j), c in self.state_material_balances.items():
                    sf = iscale.get_scaling_factor(
                        cvol.properties_in[t].get_material_flow_terms(p, j),
                        default=1,
                        warning=True,
                    )
                    iscale.constraint_scaling_transform(c, sf)
            elif mb_type == MaterialBalanceType.componentTotal:
                for (t, j), c in self.state_material_balances.items():
                    sf = iscale.min_scaling_factor(
                        [
                            cvol.properties_in[t].get_material_flow_terms(p, j)
                            for p in phase_list
                            if (p, j) in phase_component_set
                        ]
                    )
                    iscale.constraint_scaling_transform(c, sf)
            else:
                # There are some other material balance types but they create
                # constraints with different names.
                _log.warning(f"Unknown material balance type {mb_type}")
