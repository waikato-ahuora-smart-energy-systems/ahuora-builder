# Pyomo core
from pyomo.environ import (
    Constraint,
    Expression,
    NonNegativeReals,
    Suffix,
    Var,
    value,
    units as UNIT,
)
from pyomo.core.base.reference import Reference
from pyomo.common.config import ConfigBlock, ConfigValue, Bool

# IDAES core
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    StateBlock,
)
from idaes.core.util import scaling
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.solvers import get_solver
from idaes.core.initialization import ModularInitializerBase

# Other
from property_packages.build_package import build_package

# Logger
import idaes.logger as idaeslog

# Typing
from typing import List


__author__ = "Ahuora Centre for Smart Energy Systems, University of Waikato, New Zealand"

# Set up logger
_log = idaeslog.getLogger(__name__)

class SteamUserInitializer(ModularInitializerBase):
    """Initializer for ``SteamUser``.

    Parameters
    ----------
    blk : SteamUser
        The unit model block to initialize.
    solver : optional
        A Pyomo/IDAES solver instance. Defaults to :func:`idaes.core.solvers.get_solver`.
    solver_options : dict, optional
        Options to set on the solver, e.g. tolerances.
    outlvl : int, optional
        IDAES log level (e.g. :data:`idaes.logger.WARNING`).

    Returns
    -------
    pyomo.opt.results.results_.SolverResults
        Result from the final solve.
    """

    def initialize(self, blk, **kwargs):
        # --- Solver setup
        solver = kwargs.get("solver", None) or get_solver()
        solver_options = kwargs.get("solver_options", {})
        for k, v in solver_options.items():
            solver.options[k] = v

        outlvl = kwargs.get("outlvl", idaeslog.WARNING)
        log = idaeslog.getLogger(__name__)

        # --- Time index
        t0 = blk.flowsheet().time.first()

        # --- 1) Initialize inlet state blocks
        inlet_blocks = list(blk.inlet_blocks)
        if blk.config.has_desuperheating:
            inlet_steam, inlet_water = inlet_blocks
            inlet_water[t0].pressure.set_value(
                inlet_steam[t0].pressure
            )
            inlet_water[t0].enth_mol.set_value(
                blk.water.htpx(
                    blk.bfw_temperature[t0], 
                    inlet_water[t0].pressure,
                ) 
            )
        else:
            inlet_steam = inlet_blocks[0]

        for sb in inlet_blocks:
            if hasattr(sb, "initialize"):
                sb.initialize(outlvl=outlvl)

        # --- 2) Seed satuarate inlet-related state
        if blk.config.has_desuperheating:
            sb = blk._int_inlet_sat_vap_state
            sb[t0].pressure.set_value(
                inlet_steam[t0].pressure
            )
            sb[t0].enth_mol.set_value(
                inlet_steam[t0].enth_mol_sat_phase["Vap"]
            )
            if hasattr(sb, "initialize"):
                sb.initialize(outlvl=outlvl)

        # --- 3) Aggregate inlet info for seeding mixed state block
        ms = blk._int_mixed_inlet_state
        ms[t0].pressure.set_value(
            inlet_steam[t0].pressure
        )
        if blk.config.has_desuperheating:
            if value(blk.deltaT_superheat[t0]) > 0:
                ms[t0].enth_mol.set_value(
                    blk.water.htpx(
                        T=blk._int_inlet_sat_vap_state[t0].temperature + blk.deltaT_superheat[t0],
                        p=inlet_steam[t0].pressure,
                    )            
                )
            else:
                ms[t0].enth_mol.set_value(
                    blk._int_inlet_sat_vap_state[t0].enth_mol         
                )     

            if abs(value(ms[t0].enth_mol - inlet_water[t0].enth_mol)) > 0:
                ms[t0].flow_mol.set_value(
                    inlet_steam[t0].flow_mol * (inlet_steam[t0].enth_mol - ms[t0].enth_mol) / (ms[t0].enth_mol - inlet_water[t0].enth_mol)
                )
            else:
                ms[t0].flow_mol.set_value(
                    inlet_steam[t0].flow_mol
                )
        else:
            ms[t0].enth_mol.set_value(
                inlet_steam[t0].enth_mol         
            )            
            ms[t0].flow_mol.set_value(
                inlet_steam[t0].flow_mol
            )

        if hasattr(ms, "initialize"):
            ms.initialize(outlvl=outlvl)

        # --- 4) Seed outlet-related states
        # Condensate after heating (internal)
        ios = blk._int_outlet_cond_state
        ios[t0].flow_mol.set_value(
            ms[t0].flow_mol
        )
        ios[t0].pressure.set_value(
            ms[t0].pressure - blk.pressure_loss[t0]
        )
        ios[t0].enth_mol.set_value(
            blk.water.htpx(
                T=blk._int_outlet_cond_state[t0].temperature_sat - blk.deltaT_subcool[t0],
                p=ios[t0].pressure,
            )
        )
        if hasattr(ios, "initialize"):
            ios.initialize(outlvl=outlvl)

        # --- 5) Seed satuarate outlet-related state
        sb = blk._int_outlet_sat_liq_state
        sb[t0].pressure.set_value(
            ios[t0].pressure
        )
        sb[t0].enth_mol.set_value(
            ios[t0].enth_mol_sat_phase["Vap"]
        )
        if hasattr(sb, "initialize"):
            sb.initialize(outlvl=outlvl)           

        # --- 5) Seed external outlets
        # Return line properties at user-specified temperature
        ors = blk.outlet_return_state
        ors[t0].flow_mol.set_value(ms[t0].flow_mol * value(blk.cond_return_rate[t0]))
        ors[t0].pressure.set_value(
            ms[t0].pressure - blk.pressure_loss[t0]
        )
        ors[t0].enth_mol.set_value(
            blk.water.htpx(
                T=blk.cond_return_temperature[t0], 
                p=ors[t0].pressure,
            )
        )
        if hasattr(ors, "initialize"):
            ors.initialize(outlvl=outlvl)

        # --- 6) Drain/blowdown outlet uses reference enthalpy (fixed later in build)
        # Flow/pressure will be solved by constraints
        if hasattr(blk.outlet_drain_state, "initialize"):
            blk.outlet_drain_state.initialize(outlvl=outlvl)

        # --- 7) Solve
        res = solver.solve(blk, tee=False)
        log.info(f"SteamUser init status: {res.solver.termination_condition}")
        return res

def _make_config_block(config):
    """Declare configuration options for the SteamUser unit.

    Declares property package references and integer counts for inlets and outlets.

    Args:
        config (ConfigBlock): The mutable configuration block to populate.
    """

    config.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
        ),
    )
    config.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
        ),
    )
    config.declare(
        "has_desuperheating",
        ConfigValue(
            default=False,
            domain=Bool,
            description="If true, include desuperheating prior to use as process heat. " \
            "Adds the state variable of the degree of superheat after desuperheating. Default: 0.",
        ),
    )    

@declare_process_block_class("SteamUser")
class SteamUserData(UnitModelBlockData):
    """Steam user unit operation.

    The SteamUser aggregates thermal loads from multiple sub-users (heaters) within a site.
    Desuperheating the flow is optional. Heat loss and pressure loss may be defined. 
    A mixed (intermediate) states are used for balances.

    Key features:
        - Material, energy, and momentum balances around the user
        - Optional desuperheating step prior to process use
        - User-specified condensate return rate and temperature
        - Optional heat and pressure losses.

    Attributes:
        inlet_list (list[str]): Names for inlet ports.
        outlet_list (list[str]): Names for outlet ports (incl. condensate/ and vent).
        inlet_blocks (list): StateBlocks for all inlets.
        outlet_blocks (list): StateBlocks for all outlets.
        _int_mixed_inlet_state: Intermediate mixture StateBlock.
        
        heat_loss (Var): Heat loss from the header (W).
        pressure_loss (Var): Pressure drop from inlet minimum to mixed state (Pa).
        bfw_flow_mol (Var): Required inlet boiler feed water flow for desuperheating (mol/s).
    """

    default_initializer=SteamUserInitializer
    CONFIG = UnitModelBlockData.CONFIG()
    _make_config_block(CONFIG)

    def build(self) -> None:
        """Build the unit model structure (ports, states, constraints)."""
        # 1. Inherit standard UnitModelBlockData properties and functions
        super().build()

        # 2. Validate input parameters are valid
        self._validate_model_config()

        # 3. Create lists of ports with state blocks to add
        self.inlet_list = self._create_inlet_port_name_list()
        self.outlet_list = self._create_outlet_port_name_list()

        # 4. Declare ports, state blocks and state property bounds 
        self.inlet_blocks = self._add_ports_with_state_blocks(
            stream_list=self.inlet_list,
            is_inlet=True,
            has_phase_equilibrium=False,
            is_defined_state=True,
        )
        self.outlet_blocks = self._add_ports_with_state_blocks(
            stream_list=self.outlet_list,
            is_inlet=False,
            has_phase_equilibrium=False,
            is_defined_state=False
        )
        self._internal_blocks = self._add_internal_state_blocks()
        self._ref_enth = self._add_environmental_reference_enth()
        self._add_bounds_to_state_properties()
        
        # 4. Declare references, variables and expressions for external and internal use
        self._create_references()
        self._create_variables()
        self._create_expressions()

        # 5. Set balance equations
        self._add_material_balances()
        self._add_energy_balances()
        self._add_momentum_balances()
        self._add_additional_constraints()

        # 6. Other
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

    # ------------------------------------------------------------------
    # Helpers & construction utilities
    # ------------------------------------------------------------------
    def _validate_model_config(self) -> bool:
        """Validate configuration for inlet and outlet counts.

        Raises:
            ValueError: If ``property_package is None``.
        """
        if self.config.property_package is None:
            raise ValueError("SteamUser: Property package not defined.")
        return True

    def _create_inlet_port_name_list(self) -> List[str]:
        """Build ordered inlet port names.

        Returns:
            list[str]: Names
        """
        
        return (
            [
                "inlet_steam", "inlet_water"
            ] 
            if self.config.has_desuperheating else
            [
                "inlet_steam"
            ]
        )            

    def _create_outlet_port_name_list(self) -> List[str]:
        """Build ordered outlet port names.

        Returns:
            list[str]: Names 
        """        
        return [
            "outlet_return", 
            "outlet_drain",
        ]

    def _add_ports_with_state_blocks(self, 
                                     stream_list: List[str], 
                                     is_inlet: List[str], 
                                     has_phase_equilibrium: bool=False,
                                     is_defined_state: bool=None,
                                     ) -> List[StateBlock]:
        """Construct StateBlocks and expose them as ports.

        Creates a StateBlock per named stream and attaches a corresponding inlet or
        outlet Port. Inlet blocks are defined states; outlet blocks are calculated states.

        Args:
            stream_list (list[str]): Port/StateBlock base names to create.
            is_inlet (bool): If True, create inlet ports with ``defined_state=True``;
                otherwise create outlet ports with ``defined_state=False``.
            has_phase_equilibrium (bool)

        Returns:
            list: The created StateBlocks, in the same order as ``stream_list``.
        """
        # Create empty list to hold StateBlocks for return
        state_block_ls = []

        # Setup StateBlock argument dict
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = has_phase_equilibrium
        if is_defined_state == None:
            tmp_dict["defined_state"] = True if is_inlet else False
        else:
            tmp_dict["defined_state"] = is_defined_state

        # Create an instance of StateBlock for all streams
        for s in stream_list:
            sb = self.config.property_package.build_state_block(
                self.flowsheet().time, doc=f"Thermophysical properties at {s}", **tmp_dict
            )
            setattr(
                self, s + "_state", 
                sb
            )
            state_block_ls.append(sb)
            add_fn = self.add_inlet_port if is_inlet else self.add_outlet_port
            add_fn(
                name=s,
                block=sb,
            )

        return state_block_ls

    def _add_internal_state_blocks(self) -> List[StateBlock]:
        """Create the intermediate StateBlock(s)."""
        # The _int_outlet_cond_state:
        #     - Is not a defined state (solved from balances).
        #     - Represents the state of the condensate after delivering process heating.
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["defined_state"] = False

        self._int_outlet_cond_state = self.config.property_package.build_state_block(
            self.flowsheet().time, 
            doc="Thermophysical properties of condensate after process heating.", 
            **tmp_dict
        )
        # The _int_mixed_inlet_state:
        #     - Has phase equilibrium enabled.
        #     - Is not a defined state (solved from balances).
        #     - Always exists even when not desuperheating
        tmp_dict["has_phase_equilibrium"] = True
        tmp_dict["defined_state"] = False            
        self._int_mixed_inlet_state = self.config.property_package.build_state_block(
            self.flowsheet().time, 
            doc="Thermophysical properties internal mixed inlet state after desuperheating (if applicable).", 
            **tmp_dict
        )
        # The _int_outlet_sat_liq_state:
        #     - Has phase equilibrium enabled.
        #     - Is not a defined state (solved from balances).
        #     - Always exists even when not desuperheating
        tmp_dict["has_phase_equilibrium"] = True
        tmp_dict["defined_state"] = False            
        self._int_outlet_sat_liq_state = self.config.property_package.build_state_block(
            self.flowsheet().time, 
            doc="Thermophysical properties internal mixed saturate state.", 
            **tmp_dict
        )
        self._int_outlet_sat_liq_state[:].flow_mol.fix(1)    

        if self.config.has_desuperheating:
            # The _int_inlet_sat_vap_state:
            #     - Has phase equilibrium enabled.
            #     - Is not a defined state (solved from balances).
            #     - Only exists when desuperheating
            tmp_dict["has_phase_equilibrium"] = True
            tmp_dict["defined_state"] = False            
            self._int_inlet_sat_vap_state = self.config.property_package.build_state_block(
                self.flowsheet().time, 
                doc="Thermophysical properties internal mixed saturate state.", 
                **tmp_dict
            )
            self._int_inlet_sat_vap_state[:].flow_mol.fix(1)

        return [
            self._int_mixed_inlet_state,
            self._int_outlet_cond_state,
            self._int_outlet_sat_liq_state,
            self._int_inlet_sat_vap_state,
        ] if self.config.has_desuperheating else [
            self._int_mixed_inlet_state,
            self._int_outlet_cond_state,
            self._int_outlet_sat_liq_state,
        ]      
    
    def _add_environmental_reference_enth(self) -> None:
        """Create a helper to compute reference enthalpy at 15°C, 1 atm (water)."""
        self.water = build_package("helmholtz", ["water"], ["Liq"])
        return self.water.htpx(
            (15 + 273.15) * UNIT.K, 
            101325 * UNIT.Pa
        )

    def _add_bounds_to_state_properties(self) -> None:
        """Add lower and/or upper bounds to state properties.

        - Set nonnegativity lower bounds on all inlet/intermediate/outlet flows.
        """
        for sb in (self.inlet_blocks + self.outlet_blocks + self._internal_blocks):
            for t in sb:
                sb[t].flow_mol.setlb(0.0)     

    def _create_references(self) -> None:
        """Create convenient References.

        Creates references to _int_mixed_inlet_state properties:
            - ``bfw_temperature`` 
            - ``bfw_flow_mass``
            - ``bfw_flow_mol``
        """
        # Read only variables, only applicable if desuperheating is active
        if self.config.has_desuperheating:
            self.bfw_flow_mass = Reference(
                self.inlet_water_state[:].flow_mass
            )
            self.bfw_flow_mol = Reference(
                self.inlet_water_state[:].flow_mol
            )
            self.inlet_water_state[:].flow_mol.unfix()  

    def _create_variables(self) -> None:
        """Declare decision/parameter variables for the unit.

        Creates:
            - ``heat_demand`` 
            - ``cond_return_rate`` 
            - ``cond_return_temperature`` 
            - ``deltaT_subcool`` 
            - ``heat_loss`` 
            - ``pressure_loss`` 
            If desuperheating:
                - ``bfw_temperature``  
                - ``deltaT_superheat``      
        """
        # Get units consistent with the property package
        units_meta = self.config.property_package.get_metadata()

        # Calculated: Process heat demand (kW) — user typically fixes
        self.heat_demand = Var(
            self.flowsheet().time, 
            domain=NonNegativeReals,
            doc="Process heat demand of the users. Default: 0 kW.",
            units=units_meta.get_derived_units("power"),
        )
        self.heat_demand[:].set_value(
            0 # Default value
        )
        # User defined: Fraction of total condensate that returns to boiler (dimensionless)
        self.cond_return_rate = Var(
            self.flowsheet().time, 
            domain=NonNegativeReals,
            bounds=(0,1),
            doc="Fraction of condensate returned to the boiler. Default: 0.7."
        )
        self.cond_return_rate.fix(
            0.7 # Default value
        )
        # User defined: Condensate return temperature (degC or K)
        self.cond_return_temperature = Var(
            self.flowsheet().time, 
            domain=NonNegativeReals,
            doc="Temperature at which the condensate returns to the boiler. Default: 80 degC.",
            units=units_meta.get_derived_units("temperature"),
        )
        self.cond_return_temperature.fix(
            (80 + 273.15) * UNIT.K # Default fixed value
        )
        # User defined: Subcooling target delta T for condensate after process heating (K)
        self.deltaT_subcool = Var(
            self.flowsheet().time, 
            domain=NonNegativeReals,
            doc="The target amount of subcooling of the condensate after process heating. Default: 0 K.",
            units=units_meta.get_derived_units("temperature"),
        )
        self.deltaT_subcool.fix(
            0 # Default fixed value
        )
        # User defined: Heat and pressure losses
        self.heat_loss = Var(
            self.flowsheet().time,
            domain=NonNegativeReals,
            doc="Heat loss. Default: 0 kW.",
            units=units_meta.get_derived_units("power")
        )
        self.heat_loss.fix(
            0 # Default fixed value
        )        
        self.pressure_loss = Var(
            self.flowsheet().time,
            domain=NonNegativeReals,
            doc="Pressure loss. Default: 0 Pa.",
            units=units_meta.get_derived_units("pressure")
        )
        self.pressure_loss.fix(
            0 # Default fixed value
        )   
        
        # User defined when desuperheating is active, otherwise do not show
        if self.config.has_desuperheating:
            # User defined: Boiler feed water temperature entering the desuperheater
            self.bfw_temperature = Var(
                self.flowsheet().time, 
                domain=NonNegativeReals,
                doc="The target amount of subcooling of the condensate after process heating. Default: 0 K.",
                units=units_meta.get_derived_units("temperature"),
            )
            self.bfw_temperature.fix(
                (110 + 273.15) * UNIT.K # Default fixed value
            )
            # User defined: Target degree of superheat at the outlet of the desuperheater
            self.deltaT_superheat = Var(
                self.flowsheet().time, 
                domain=NonNegativeReals,
                doc="The target amount of superheat present in the steam after desuperheating before use. Default: 0 K.",
                units=units_meta.get_derived_units("temperature"),
            )
            self.deltaT_superheat.fix(
                0 # Default fixed value
            )

    def _create_expressions(self) -> None:
        """Create helper Expressions.

        Creates:
            - ``energy_lost`` 
        """
        # Calculated, always show
        self.energy_lost = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                b._int_outlet_cond_state[t].flow_mol
                * (b._int_outlet_cond_state[t].enth_mol - b._ref_enth)
                - 
                b.outlet_return_state[t].flow_mol
                * (b.outlet_return_state[t].enth_mol   - b._ref_enth)
            ),
            doc="Energy lost from condensate cooling and condensate to drain.",
        )

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------
    def _add_material_balances(self) -> None:
        """Material balance equations summary.

        Balances / Constraints:
            - ``overall_material_balance``
            - ``condensate_return_material_eq``
            - ``intermediate_material_balance_post_heating``
            - ``intermediate_material_balance_pre_heating``
        """
        @self.Constraint(
                self.flowsheet().time, 
                doc="Overall material balance",
                )
        def overall_material_balance(b, t):
            return (
                sum(
                    o[t].flow_mol
                    for o in b.outlet_blocks
                )
                == 
                sum(
                    i[t].flow_mol
                    for i in b.inlet_blocks
                )
            )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Intermediate material balance",
                )
        def intermediate_material_balance_pre_heating(b, t):
            return (
                b._int_mixed_inlet_state[t].flow_mol
                ==
                sum(
                    i[t].flow_mol
                    for i in b.inlet_blocks
                ) 
            )             
        @self.Constraint(
                self.flowsheet().time, 
                doc="Intermediate material balance",
                )
        def intermediate_material_balance_post_heating(b, t):
            return (
                b._int_outlet_cond_state[t].flow_mol
                ==
                sum(
                    i[t].flow_mol
                    for i in b.inlet_blocks
                ) 
            )        
        @self.Constraint(
                self.flowsheet().time, 
                doc="Condensate return material equation",
                )
        def condensate_return_material_eq(b, t):
            return (
                b.outlet_return_state[t].flow_mol 
                ==
                sum(
                    i[t].flow_mol
                    for i in b.inlet_blocks
                ) 
                *
                b.cond_return_rate[t]
            )
        
    def _add_energy_balances(self) -> None:
        """Energy balance equations summary.

        Balances / Constraints:
            - ``mixing_energy_balance``
            - ``heating_energy_balance``
        """
        @self.Constraint(
                self.flowsheet().time, 
                doc="Inlet mixing energy balance",
                )
        def mixing_energy_balance(b, t):
            return (
                b._int_mixed_inlet_state[t].flow_mol * b._int_mixed_inlet_state[t].enth_mol
                ==
                sum(
                    i[t].flow_mol * i[t].enth_mol
                    for i in b.inlet_blocks
                )
            )
        @self.Constraint(
                self.flowsheet().time,
                doc="Process heating energy balance",
                )
        def heating_energy_balance(b, t):
            return (
                b._int_mixed_inlet_state[t].flow_mol * b._int_mixed_inlet_state[t].enth_mol
                == 
                b._int_outlet_cond_state[t].flow_mol * b._int_outlet_cond_state[t].enth_mol
                +
                b.heat_loss[t]
                +
                b.heat_demand[t]
            )
        self.outlet_drain_state[:].enth_mol.fix( 
            value(
                self._ref_enth
            )
        )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Saturated liquid enthalpy",
                )            
        def saturated_liq_enthalpy_eq(b, t):
            return (
                b._int_outlet_sat_liq_state[t].enth_mol_sat_phase["Liq"]
                ==
                b._int_outlet_sat_liq_state[t].enth_mol
            )
        if self.config.has_desuperheating:
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Saturated vapour enthalpy",
                    )            
            def saturated_vap_enthalpy_eq(b, t):
                return (
                    b._int_inlet_sat_vap_state[t].enth_mol_sat_phase["Vap"]
                    ==
                    b._int_inlet_sat_vap_state[t].enth_mol
                )           

    def _add_momentum_balances(self) -> None:
        """Momentum balance equations summary.

        Balances / Constraints:
            - ``mixing_momentum_balance``
            - ``heating_momentum_balance``
            - ``outlet_momentum_balance``
            If desuperheating:
                - ``intlet_water_momentum_balance``            
        """
        @self.Constraint(
                self.flowsheet().time,
                doc="Momentum equalities",
                )
        def mixing_momentum_balance(b, t):
            return (
                b.inlet_steam_state[t].pressure
                ==
                b._int_mixed_inlet_state[t].pressure 
            )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Process heating momentum balance",
                )
        def heating_momentum_balance(b, t):
            return (
                b._int_mixed_inlet_state[t].pressure
                == 
                b._int_outlet_cond_state[t].pressure 
                + 
                b.pressure_loss[t]
            )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Saturated liq pressure",
                )
        def saturated_liq_pressure_eq(b, t):
            return (
                b._int_outlet_sat_liq_state[t].pressure
                == 
                b._int_outlet_cond_state[t].pressure 
            )        
        @self.Constraint(
                self.flowsheet().time, 
                doc="Outlet momentum equality",
                )
        def outlet_momentum_balance(b, t):
            return (
                b.outlet_return_state[t].pressure
                == 
                b.outlet_drain_state[t].pressure
            )
        self.outlet_return_state[:].pressure.fix(
            101325 * UNIT.Pa # Fixed value, hidden from the user
        )
        if self.config.has_desuperheating:
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Inlet water momentum balance",
                    )
            def intlet_water_momentum_balance(b, t):
                return (
                    b.inlet_water_state[t].pressure
                    == 
                    b.inlet_steam_state[t].pressure 
                )
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Saturated vapour pressure",
                    )            
            def saturated_vap_pressure_eq(b, t):
                return (
                    b.inlet_steam_state[t].pressure
                    ==
                    b._int_inlet_sat_vap_state[t].pressure
                )                 

    def _add_additional_constraints(self) -> None:
        """Add auxiliary constraints and bounds.

        Constraints:
            - ``condensate_temperature_eq``
            - ``subcooling_temperature_eq``
            If desuperheating:
                - ``desuperheating_mixed_temperature_eq``    
        """
        @self.Constraint(
                self.flowsheet().time, 
                doc="Condensate return temperature",
                )
        def condensate_temperature_eq(b, t):
            return (
                b.outlet_return_state[t].temperature
                == 
                b.cond_return_temperature[t]
            )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Subcool temperature",
                )
        def subcooling_temperature_eq(b, t):
            return (
                b._int_outlet_cond_state[t].temperature
                == 
                b._int_outlet_sat_liq_state[t].temperature - b.deltaT_subcool[t]
            )
        if self.config.has_desuperheating:
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Inlet water temperature",
                    )
            def inlet_water_temperature_eq(b, t):
                return (
                    b.inlet_water_state[t].temperature
                    == 
                    b.bfw_temperature[t]
                )            
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Mixed temperature after desuperheating",
                    )
            def desuperheating_mixed_temperature_eq(b, t):
                return (
                    b._int_mixed_inlet_state[t].temperature 
                    ==
                    b._int_inlet_sat_vap_state[t].temperature + b.deltaT_superheat[t]
                )         

    def calculate_scaling_factors(self):
        """Assign scaling factors to improve numerical conditioning.

        Sets scaling factors for performance and auxiliary variables. 
        """
        super().calculate_scaling_factors()
        scaling.set_scaling_factor(self.heat_loss, 1e-6) # kW scale
        scaling.set_scaling_factor(self.pressure_loss, 1e-6) # Pa scale

    def _get_stream_table_contents(self, time_point=0):
        """Create a stream table for all inlets and outlets.

        Args:
            time_point (int | float): Time index at which to extract stream data.

        Returns:
            pandas.DataFrame: A tabular view suitable for reporting via
            ``create_stream_table_dataframe``.
        """
        io_dict = {}

        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name)

        for outlet_name in self.outlet_list:
            io_dict[outlet_name] = getattr(self, outlet_name)

        return create_stream_table_dataframe(io_dict, time_point=time_point)

    def _get_performance_contents(self, time_point=0):
        """Collect performance variables for reporting.

        Args:
            time_point (int | float): Time index at which to report values.

        Returns:
            dict: Mapping used by IDAES reporters, containing human-friendly labels
            to Vars/References (e.g., heat/pressure loss, mixed-state properties).
        """
        perf = {
            "vars": {
                "Heat demand [W]": self.heat_demand[time_point],
                "Degree of subcooling target [K]": self.deltaT_subcool[time_point],
                "Heat loss [W]": self.heat_loss[time_point],
                "Pressure loss [Pa]": self.pressure_loss[time_point],
                "Condensate return rate [-]:": self.cond_return_rate[time_point],
                "Condensate return temperature [degC]": UNIT.convert_temp_K_to_C(self.cond_return_temperature[time_point]),
            },
            "exprs": {
                "Energy lost to return network [W]": self.energy_lost[time_point],
            },
        }
        if self.config.has_desuperheating:
            perf["vars"].update(
                {
                    "BFW temperature [K]": self.bfw_temperature[time_point],
                    "Degree of superheat target [K]": self.deltaT_superheat[time_point],
                }
            )
        return perf

    def initialize(self, *args, **kwargs):
        """Initialize the SteamUser unit using :class:`SteamUserInitializer`.

        Args:
            *args: Forwarded to ``SteamUserInitializer.initialize``.
            **kwargs: Forwarded to ``SteamUserInitializer.initialize`` (e.g., solver, options).

        Returns:
            pyomo.opt.results.results_.SolverResults: Results from the initializer's solve.
        """
        init = SteamUserInitializer()
        return init.initialize(self, *args, **kwargs)
