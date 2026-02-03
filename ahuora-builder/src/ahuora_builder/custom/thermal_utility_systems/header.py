# Pyomo core
import pyomo.environ as pyo
from pyomo.environ import (
    Constraint,
    Expression,
    Param,
    PositiveReals,
    RangeSet,
    Suffix,
    Var,
    value,
    units as UNIT,
)
from pyomo.core.base.reference import Reference
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool

# IDAES core
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    StateBlock,
)
from idaes.core.util import scaling
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.math import smooth_min, smooth_max
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.solvers import get_solver
from idaes.core.initialization import ModularInitializerBase
from idaes.core.util.model_statistics import degrees_of_freedom

# Logger
import idaes.logger as idaeslog

# Typing
from typing import List


__author__ = "Ahuora Centre for Smart Energy Systems, University of Waikato, New Zealand"

# Set up logger
_log = idaeslog.getLogger(__name__)

class SimpleHeaderInitializer(ModularInitializerBase):
    """Initialize a Header unit block with staged seeding and solves.

    This routine performs a two-stage initialization:
    1) Seed inlet and internal state variables, relax selected constraints, and
       perform a first solve.
    2) Reactivate/tighten constraints and perform a second solve.

    Args:
        blk: The Header unit model block to initialize.
        **kwargs: Optional keyword arguments:
            solver: A Pyomo/IDAES solver object. If not provided, uses ``get_solver()``.
            solver_options (dict): Options to set on the solver, e.g. tolerances.
            outlvl: IDAES log level (e.g., ``idaeslog.WARNING``).

    Returns:
        pyomo.opt.results.results_.SolverResults: The result object from the final solve.

    Notes:
        - Inlet state blocks are initialized via their own ``initialize`` if available.
        - Mixed state is seeded from inlet totals/minimums (pressure) and average
          enthalpy; works with temperature- or enthalpy-based property packages.
        - Temporary seeds/relaxations are undone, leaving original DOF intact.
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
        if len(inlet_blocks) < 1:
            raise ValueError("No inlet added to header.")
        
        for sb in inlet_blocks:
            if hasattr(sb, "initialize"):
                sb.initialize(outlvl=outlvl)

        # --- 2) Aggregate inlet info for seeding mixed state block
        F_mixed = sum(
            value(sb[t0].flow_mol) 
            for sb in inlet_blocks
        )
        P_mixed = min(
            value(sb[t0].pressure) 
            for sb in inlet_blocks
        )
        E_mixed = sum(
            value(sb[t0].flow_mol * sb[t0].enth_mol, 0.0) 
            for sb in inlet_blocks
        )
        if F_mixed > 0:
            h_mixed = E_mixed / F_mixed
        else:
            # Seed from the first inlet’s enthalpy (no double subscripting)
            first_inlet = inlet_blocks[0]
            h_mixed = value(first_inlet[t0].enth_mol)

        # --- 3) Seed mixed_state: flow, pressure, enthalpy
        ms = blk.mixed_state
        ms[t0].flow_mol.set_value(
            F_mixed
        )
        ms[t0].pressure.set_value(
            P_mixed
        )
        ms[t0].enth_mol.set_value(
            h_mixed
        ) 
        ms.initialize(outlvl=outlvl)     

        # --- 4) Seed outlet_states with pressure, enthalpy
        flow_undefined = []
        defined_flow = 0
        for sb in blk.outlet_blocks:
            sb[t0].pressure.set_value(
                value(ms[t0].pressure)
            )
            sb[t0].enth_mol.set_value(
                value(ms[t0].enth_mol)
            )
            if sb in [blk.outlet_condensate_state, blk.outlet_vent_state]:
                sb[t0].flow_mol.set_value(
                    0.0
                )
            else:
                if value(sb[t0].flow_mol, exception=False) is None:
                    flow_undefined.append(sb) 
                else:
                    defined_flow += value(sb[t0].flow_mol)
        
        tot_undefined_flow = max(sum(value(sb[t0].flow_mol) for sb in blk.inlet_blocks) - defined_flow, 0)
        for sb in flow_undefined:
            sb[t0].flow_mol.set_value(
                tot_undefined_flow / len(flow_undefined)
            )

        for sb in blk.outlet_blocks:        
            sb.initialize(outlvl=outlvl)

        res2 = solver.solve(blk, tee=False)
        log.info(f"Header init status: {res2.solver.termination_condition}")

        return res2

def _make_config_block(config):
    """Declare configuration options for the Header unit.

    Declares property package references and integer counts for inlets and outlets.

    Args:
        config (ConfigBlock): The mutable configuration block to populate.

    Raises:
        ValueError: If invalid option values are provided by the caller (via IDAES).
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
        "num_inlets",
        ConfigValue(
            default=1,
            domain=In(list(range(1, 100))),
            description="Number of utility providers at inlets.",
        ),
    )
    config.declare(
        "num_outlets",
        ConfigValue(
            default=1,
            domain=In(list(range(0, 100))),
            description="Number of utility users at outlets." \
            "Excludes outlets associated with condensate and vent flows.",
        ),
    )
    config.declare(
        "is_liquid_header",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Flag for selecting liquid or vapour (including steam and other gases).",
        ),
    )
@declare_process_block_class("simple_header")
class SimpleHeaderData(UnitModelBlockData):
    """Thermal utility header unit operation.

    The Header aggregates multiple inlet providers and distributes utility to
    multiple users, with optional venting, condensate removal (or liquid overflow), heat loss, and
    pressure loss. A mixed (intermediate) state is used for balances and
    pressure/enthalpy coupling across outlets.

    Key features:
        - Material, energy, and momentum balances with smooth min/max functions.
        - Vapour/liquid equilibrium calculation for mixed state.
        - Shared mixed enthalpy across outlets of the same phase.
        - Computed excess flow from an overall flow balance.
        - Optional heat and pressure losses.

    Attributes:
        inlet_list (list[str]): Names for inlet ports.
        outlet_list (list[str]): Names for outlet ports (incl. condensate/ and vent).
        inlet_blocks (list): StateBlocks for all inlets.
        outlet_blocks (list): StateBlocks for all outlets.
        mixed_state: Intermediate mixture StateBlock.
        heat_loss (Var): Heat loss from the header (W).
        pressure_loss (Var): Pressure drop from inlet minimum to mixed state (Pa).
        makeup_flow_mol (Var): Required inlet makeup molar flow (mol/s).
    """

    default_initializer=SimpleHeaderInitializer
    CONFIG = UnitModelBlockData.CONFIG()
    _make_config_block(CONFIG)

    def build(self) -> None:
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
        self._add_bounds_to_state_properties()
        self._outlet_supply_blocks = self._create_custom_state_lists()
        
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
        self.split_flow = self._create_flow_map_references()

    def _validate_model_config(self) -> bool:
        """Validate configuration for inlet and outlet counts.

        Raises:
            ValueError: If ``num_inlets < 1`` or ``num_outlets < 1``.
        """
        if self.config.num_inlets < 1:
            raise ValueError("Header requires at least one provider (num_inlets >= 1).")
        if self.config.num_outlets < 1:
            raise ValueError("Header requires at least one user (num_outlets >= 1).")
        return True

    def _create_inlet_port_name_list(self) -> List[str]:
        """Build ordered inlet port names.

        Returns:
            list[str]: Names ``["inlet_1", ..., "inlet_N"]`` based on ``num_inlets``.
        """
        return [
            f"inlet_{i+1}" for i in range(self.config.num_inlets)
        ]

    def _create_outlet_port_name_list(self) -> List[str]:
        """Build ordered outlet port names.

        Returns:
            list[str]: Names ``["outlet_1", ..., "outlet_n", "outlet_condensate", "outlet_vent"]``.
        """        
        return [
            f"outlet_{i+1}" for i in range(self.config.num_outlets)
        ] + ["outlet_condensate"] + ["outlet_vent"]

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
        """Create the intermediate (mixed) StateBlock.

        The mixed state:
            - Has phase equilibrium enabled.
            - Is not a defined state (solved from balances).
        """
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = True
        tmp_dict["defined_state"] = False

        self.mixed_state = self.config.property_package.build_state_block(
            self.flowsheet().time, 
            doc=f"Thermophysical properties at intermediate mixed state.", 
            **tmp_dict
        )
        return [
            self.mixed_state
        ]

    def _add_bounds_to_state_properties(self) -> None:
        """Add lower and/or upper bounds to state properties.

        - Set nonnegativity lower bounds on all inlet/outlet molar flows.
        """
        for sb in (self.inlet_blocks + self.outlet_blocks):
            for t in sb:
                sb[t].flow_mol.setlb(0.0)     
    
    def _create_custom_state_lists(self) -> List[StateBlock]:
        """Partition outlet names into vapour outlets and capture their StateBlocks.

        Populates:
            - ``_outlet_supply_list``: Outlet names excluding condensate and vent.
            - ``_outlet_supply_blocks``: Corresponding StateBlocks.
        """        
        self._outlet_supply_list = [
            v for v in self.outlet_list 
            if not v in ["outlet_condensate", "outlet_vent"]
        ]
        return [
            getattr(self, n + "_state") 
            for n in self._outlet_supply_list
        ]

    def _create_references(self) -> None:
        """Create convenient References.

        Creates references to mixed_state properties:
            - ``total_flow_mol`` 
            - ``total_flow_mass``
            - ``pressure`` 
            - ``temperature`` 
            - ``enth_mol`` 
            - ``enth_mass`` 
            - ``vapor_frac``
        """
        self.total_flow_mol = Reference(
            self.mixed_state[:].flow_mol
        )
        self.total_flow_mass = Reference(
            self.mixed_state[:].flow_mass
        )
        self.pressure = Reference(
            self.mixed_state[:].pressure
            )
        self.temperature = Reference(
            self.mixed_state[:].temperature
        )
        self.enth_mol = Reference(
            self.mixed_state[:].enth_mol
            )
        self.enth_mass = Reference(
            self.mixed_state[:].enth_mass
            )
        self.vapor_frac = Reference(
            self.mixed_state[:].vapor_frac
        )

    def _create_variables(self) -> None:
        """Create required variables.

        Creates:
            - ``heat_loss`` (W)
            - ``pressure_loss`` (Pa)
        """
        self.heat_loss = Var(
            self.flowsheet().time, 
            initialize=0.0, 
            doc="Heat loss",
            units=UNIT.W
        )
        self.pressure_loss = Var(
            self.flowsheet().time, 
            initialize=0.0, 
            doc="Pressure loss",
            units=UNIT.Pa
        )                

    def _create_expressions(self) -> None:
        """Create convenient Expressions.

        Creates:
            - ``balance_flow_mol`` (mol/s)
            - ``degree_of_superheat`` (K)
            - ``makeup_flow_mol`` (mol/s)
            - ``_partial_total_flow_mol`` (mol/s): used for scaling purposes in a material balance
        """
        self.degree_of_superheat = Expression(
            self.flowsheet().time,
            rule=lambda b, t: b.temperature[t] - b.outlet_condensate_state[t].temperature
        )
        self._partial_total_flow_mol = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                sum(
                    o[t].flow_mol
                    for o in (b.inlet_blocks + b._outlet_supply_blocks)
                )                
            )
        )       
        self.balance_flow_mol = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                sum(
                    i[t].flow_mol
                    for i in b.inlet_blocks
                )
                -
                sum(
                    o[t].flow_mol
                    for o in (
                        b._outlet_supply_blocks + 
                        [
                            b.outlet_vent_state 
                            if self.config.is_liquid_header 
                            else b.outlet_condensate_state
                        ]
                    )
                )                
            )
        )        
        self.makeup_flow_mol = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                (
                    b.outlet_condensate_state[t].flow_mol 
                    if self.config.is_liquid_header 
                    else b.outlet_vent_state[t].flow_mol
                ) 
                - 
                b.balance_flow_mol[t]
            )
        )

    def _add_material_balances(self) -> None:
        """Material balance equations summary.

        Introduces:
            - ``_partial_total_flow_mol``: Sum of known inlet and vapour outlet flows,
            used for scaling a smooth vent calculation.

        Constraints:
            - ``mixed_state_material_balance``: Mixed flow equals total inlet flow.
            - ``vent_flow_balance``: Depends on the header's primary phase: liquid vs gas
                                        If gas header, smoothly enforces nonnegative vent flow.
                                        If liquid header, determines flow from mixed-state vapour fraction
            - ``condensate_flow_balance``: Depends on the header's primary phase: liquid vs gas
                                        If gas header, determines flow from mixed-state vapour fraction
                                        If liquid header, smoothly enforces nonnegative condensate flow.
        """

        @self.Constraint(
                self.flowsheet().time, 
                doc="Mixed state material balance",
                )
        def mixed_state_material_balance(b, t):
            return (
                b.mixed_state[t].flow_mol
                == 
                sum(
                    i[t].flow_mol
                    for i in b.inlet_blocks
                )
            )

        eps = 1e-5  # smoothing parameter; smaller = closer to exact max, larger = smoother
        if self.config.is_liquid_header:
            # Assigns excess liquid flow to outlet_condensate
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Condensate flow balance." \
                    "Determines the positive amount of excess flow that exits through outlet_condensate"
                    )
            def condensate_flow_balance(b, t):
                return (
                    b.outlet_condensate_state[t].flow_mol
                    == 
                    smooth_max(
                        b.balance_flow_mol[t] / (b._partial_total_flow_mol[t] + 1e-6),
                        0.0,
                        eps,
                    ) * (b._partial_total_flow_mol[t] + 1e-6)
                )
                        
            # Removes any gas/vapour from a liquid header
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Vent balance."
                    )
            def vent_flow_balance(b, t):
                return b.outlet_vent_state[t].flow_mol == (
                    b.mixed_state[t].flow_mol * b.mixed_state[t].vapor_frac
                )
        else:
            # Assigns excess steam/vapour flow to outlet_vent
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Vent flow balance." \
                    "Determines the positive amount of excess flow that exits through the vent"
                    )
            def vent_flow_balance(b, t):
                return (
                    b.outlet_vent_state[t].flow_mol
                    == 
                    smooth_max(
                        b.balance_flow_mol[t] / (b._partial_total_flow_mol[t] + 1e-6),
                        0.0,
                        eps,
                    ) * (b._partial_total_flow_mol[t] + 1e-6)
                )

            # Removes any condensate/liquid from a steam/gas header
            @self.Constraint(
                    self.flowsheet().time, 
                    doc="Condensate balance."
                    )
            def condensate_flow_balance(b, t):
                return (
                    b.outlet_condensate_state[t].flow_mol 
                    ==
                    b.mixed_state[t].flow_mol * (1 - b.mixed_state[t].vapor_frac)
                )
            
    def _add_energy_balances(self) -> None:
        """Energy balance equations summary.

        Introduces:
            - ``_liq_out_enth_mol``: Shared molar enthalpy for all liquid outlets,
            including the condensate.
            - ``_vap_out_enth_mol``: Shared molar enthalpy for all vapour outlets,
            including the vent.
            
        Constraints:
            - ``inlets_to_mixed_state_energy_balance``: Inlet energy to mixed state (+ heat loss).
            - ``mixed_state_to_outlets_energy_balance``: Mixed state to all outlets.
            - ``molar_enthalpy_equality_eqn``: Common vapour enthalpy across vapour outlets and vent.
        """
        @self.Constraint(self.flowsheet().time, doc="Inlets to mixed state energy balance including heat loss")
        def inlets_to_mixed_state_energy_balance(b, t):
            return (
                b.mixed_state[t].flow_mol * b.mixed_state[t].enth_mol 
                + b.heat_loss[t]
                == 
                sum(
                    i[t].flow_mol * i[t].enth_mol
                    for i in b.inlet_blocks
                )
            )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Mixed state to outlets energy balance"
                )
        def mixed_state_to_outlets_energy_balance(b, t):
            return (
                b.mixed_state[t].enth_mol 
                * 
                sum(
                    o[t].flow_mol
                    for o in b.outlet_blocks
                )
                == 
                sum(
                    o[t].flow_mol * o[t].enth_mol
                    for o in b.outlet_blocks
                )           
            )   
        if self.config.is_liquid_header:
            self._liq_out_enth_mol = Var(
                self.flowsheet().time, 
                initialize=42.0 * 18, 
                doc="Molar enthalpy of the liquid outlets", 
                units=UNIT.J / UNIT.mol
            )
            @self.Constraint(
                self.flowsheet().time,
                self._outlet_supply_blocks + [self.outlet_condensate_state], # exclude vent outlet
                doc="All liquid outlets (incl. condensate) share a common liquid enthalpy",
                )
            def molar_enthalpy_equality_eqn(b, t, o):
                return (
                    o[t].enth_mol 
                    == 
                    b._liq_out_enth_mol[t]
                )
        else:
            self._vap_out_enth_mol = Var(
                self.flowsheet().time, 
                initialize=2700.0 * 18, 
                doc="Molar enthalpy of the vapour outlets", 
                units=UNIT.J / UNIT.mol
            )        
            @self.Constraint(
                self.flowsheet().time,
                self._outlet_supply_blocks + [self.outlet_vent_state], # exclude condensate outlet
                doc="All vapour outlets (incl. vent) share a common vapour enthalpy",
                )
            def molar_enthalpy_equality_eqn(b, t, o):
                return (
                    o[t].enth_mol 
                    == 
                    b._vap_out_enth_mol[t]
                )

    def _add_momentum_balances(self) -> None:
        """Momentum balance equations summary.

        Computes the minimum inlet pressure via a sequential smooth minimum and
        sets the mixed-state pressure to that minimum minus ``pressure_loss``,
        then enforces equality to every outlet pressure.

        Notes:
            - Uses IDAES ``smooth_min`` for differentiable minimum pressure.
            - ``_eps_pressure`` is a smoothing parameter (units of pressure).
        """
        inlet_idx = RangeSet(len(self.inlet_blocks))
        # Get units metadata
        units = self.mixed_state.params.get_metadata()
        # Add variables
        self._minimum_pressure = Var(
            self.flowsheet().time,
            inlet_idx,
            doc="Variable for calculating minimum inlet pressure",
            units=units.get_derived_units("pressure"),
        )
        self._eps_pressure = Param(
            mutable=True,
            initialize=1e-3,
            domain=PositiveReals,
            doc="Smoothing term for minimum inlet pressure",
            units=units.get_derived_units("pressure"),
        )
        # Calculate minimum inlet pressure
        @self.Constraint(
            self.flowsheet().time,
            inlet_idx,
            doc="Calculation for minimum inlet pressure",
        )
        def minimum_pressure_constraint(b, t, i):
            if i == inlet_idx.first():
                return (
                    b._minimum_pressure[t, i] 
                    == 
                    (b.inlet_blocks[i - 1][t].pressure)
                )
            else:
                return (
                    b._minimum_pressure[t, i] 
                    ==
                    smooth_min(
                        b._minimum_pressure[t, i - 1],
                        b.inlet_blocks[i - 1][t].pressure,
                        b._eps_pressure,
                    )
                )
        # Set mixed pressure to minimum inlet pressure minus any pressure loss
        @self.Constraint(
            self.flowsheet().time,
            doc="Pressure equality constraint from minimum inlet to mixed state",
        )
        def mixture_pressure(b, t):
            return (
                b.mixed_state[t].pressure 
                == 
                b._minimum_pressure[t, inlet_idx.last()] - b.pressure_loss[t]
            )
        # Set outlet pressures to mixed pressure
        @self.Constraint(
            self.flowsheet().time,
            self.outlet_blocks,
            doc="Pressure equality constraint from mixed state to outlets",
        )
        def pressure_equality_eqn(b, t, o):
            return (
                b.mixed_state[t].pressure 
                == 
                o[t].pressure
            )

    def _add_additional_constraints(self) -> None:
        """Add auxiliary constraints and bounds.
        
        - Fix vent vapour fraction to near one (near 100% vapour).
        OR
        - Fix condensate vapour fraction to a small value (near 100% liquid).
        """
        if self.config.is_liquid_header:
            @self.Constraint(self.flowsheet().time, doc="Vent vapour fraction.")
            def vent_vapour_fraction(b, t):
                return (
                    b.outlet_vent_state[t].vapor_frac 
                    == 
                    1 #1 - 1e-6
                )
        else:
            @self.Constraint(self.flowsheet().time, doc="Condensate vapour fraction.")
            def condensate_vapour_fraction(b, t):
                return (
                    b.outlet_condensate_state[t].vapor_frac 
                    == 
                    0 #1e-6
                )

    def _create_flow_map_references(self):
        """Create a two-key Reference for outlet flows over time and outlet name.

        Builds a mapping ``(t, outlet_name) -> outlet_state[t].flow_mol`` and exposes it
        as a Reference for compact access to outlet flow splits.

        Returns:
            pyomo.core.base.reference.Reference: A Reference indexed by ``(time, outlet)``.
        """
        self.outlet_idx = pyo.Set(initialize=self.outlet_list)
        # Map each (t, o) to the outlet state's flow var
        ref_map = {}
        for o in self.outlet_list:
            if o != "vent":
                outlet_state_block = getattr(self, f"{o}_state")
                for t in self.flowsheet().time:
                    ref_map[(t, o)] = outlet_state_block[t].flow_mol

        return Reference(ref_map)

    def calculate_scaling_factors(self):
        """Assign scaling factors to improve numerical conditioning.

        Sets scaling factors for performance and auxiliary variables. If present,
        also scales the shared vapour enthalpy variable ``_vap_out_enth_mol``.
        """
        super().calculate_scaling_factors()
        scaling.set_scaling_factor(self.heat_loss, 1e-6)
        scaling.set_scaling_factor(self.pressure_loss, 1e-6)
        scaling.set_scaling_factor(self.balance_flow_mol, 1e-3)
        scaling.set_scaling_factor(self._partial_total_flow_mol, 1e-3)
        if hasattr(self, "_vap_out_enth_mol"):
            scaling.set_scaling_factor(self._vap_out_enth_mol, 1e-6)

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

    def _get_performance_contents(self, time_point=0, is_full_report=True):
        """Collect performance results for reporting.

        Args:
            time_point (int | float): Time index at which to report values.
            is_full_report (bool): Flag for full or partial performance report.

        Returns:
            dict: A report of internal unit model results.
        """
        return (
            {
                "vars": {
                    "Heat Loss": self.heat_loss[time_point],
                    "Pressure Drop": self.pressure_loss[time_point],
                    "Mass Flow": self.mixed_state[time_point].flow_mass,
                    "Molar Flow": self.mixed_state[time_point].flow_mol,
                    "Balance Flow": self.balance_flow_mol[time_point],
                    "Pressure": self.mixed_state[time_point].pressure,
                    "Temperature": self.mixed_state[time_point].temperature,
                    "Degree of Superheat": self.degree_of_superheat[time_point],
                    "Vapour Fraction": self.mixed_state[time_point].vapor_frac,
                    "Mass Specific Enthalpy": self.mixed_state[time_point].enth_mass,
                    "Molar Specific Enthalpy": self.mixed_state[time_point].enth_mol,
                }
            } if is_full_report else {
                "vars": {
                    "Balance Flow": self.balance_flow_mol[time_point],
                    "Pressure": self.mixed_state[time_point].pressure,
                    "Temperature": self.mixed_state[time_point].temperature,
                    "Degree of Superheat": self.degree_of_superheat[time_point],
                }
            }

        )
            
    def initialize(self, *args, **kwargs):
        """Initialize the Header unit using :class:`SimpleHeaderInitializer`.

        Args:
            *args: Forwarded to ``SimpleHeaderInitializer.initialize``.
            **kwargs: Forwarded to ``SimpleHeaderInitializer.initialize`` (e.g., solver, options).

        Returns:
            pyomo.opt.results.results_.SolverResults: Results from the initializer's solve.
        """
        init = SimpleHeaderInitializer()
        return init.initialize(self, *args, **kwargs)
