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

class DesuperheaterInitializer(ModularInitializerBase):
    """Initializer for ``Desuperheater``.

    Parameters
    ----------
    blk : Desuperheater
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
        for sb in inlet_blocks:
            if hasattr(sb, "initialize"):
                sb.initialize(outlvl=outlvl)

        # --- 2) Seed satuarate inlet-related state
        sb = blk._int_sat_vap_state
        sb[t0].pressure.set_value(
            inlet_steam[t0].pressure
        )
        sb[t0].enth_mol.set_value(
            inlet_steam[t0].enth_mol_sat_phase["Vap"]
        )
        if hasattr(sb, "initialize"):
            sb.initialize(outlvl=outlvl)

        # --- 3) Aggregate inlet info for seeding mixed state block
        ms = blk.outlet_state
        ms[t0].pressure.set_value(
            inlet_steam[t0].pressure
        )
        if value(blk.deltaT_superheat[t0]) > 0:
            ms[t0].enth_mol.set_value(
                blk.water.htpx(
                    T=blk._int_sat_vap_state[t0].temperature + blk.deltaT_superheat[t0],
                    p=inlet_steam[t0].pressure,
                )            
            )
        else:
            ms[t0].enth_mol.set_value(
                blk._int_sat_vap_state[t0].enth_mol         
            )     

        if value(inlet_steam[t0].enth_mol) > value(ms[t0].enth_mol) > value(inlet_water[t0].enth_mol):
            ms[t0].flow_mol.set_value(
                inlet_steam[t0].flow_mol * (inlet_steam[t0].enth_mol - ms[t0].enth_mol) / (ms[t0].enth_mol - inlet_water[t0].enth_mol)
            )
        else:
            ms[t0].flow_mol.set_value(
                inlet_steam[t0].flow_mol
            )
        if hasattr(ms, "initialize"):
            ms.initialize(outlvl=outlvl)

        # --- 4) Solve
        res = solver.solve(blk, tee=False)
        log.info(f"Desuperheater init status: {res.solver.termination_condition}")
        return res

def _make_config_block(config):
    """Declare configuration options for the Desuperheater unit.

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

@declare_process_block_class("Desuperheater")
class DesuperheaterData(UnitModelBlockData):
    """Desuperheater unit operation.

    The Desuperheater injects a small water flow into a superheated steam flow to reduce superheat.
    Desuperheating is common before using steam. Heat loss and pressure loss may be defined. 
    An intermediate saturated state is used for a key reference point.

    Key features:
        - Material, energy, and momentum balances around the desuperheater
        - User-specified target amount of superheat at the exit of the desuperheater
        - Optional heat and pressure losses.

    Attributes:
        inlet_list (list[str]): Names for inlet ports.
        outlet_list (list[str]): Names for outlet ports (incl. condensate/ and vent).
        inlet_blocks (list): StateBlocks for all inlets.
        outlet_blocks (list): StateBlocks for all outlets.
        _int_sat_vap_state: Intermediate saturated vapour StateBlock.
        
    State variables:
        deltaT_superheat (Var): Target degree of superheat in the steam at the exit of the process.
        bfw_temperature (Var): Temperature of the inlet boiler feed water flow for desuperheating (degC).
        heat_loss (Var): Heat loss from the header (W).
        pressure_loss (Var): Pressure drop from inlet minimum to mixed state (Pa).
    """

    default_initializer=DesuperheaterInitializer
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
            raise ValueError("Desuperheater: Property package not defined.")
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
        )            

    def _create_outlet_port_name_list(self) -> List[str]:
        """Build ordered outlet port names.

        Returns:
            list[str]: Names 
        """        
        return [
            "outlet",
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
        # The _int_sat_vap_state:
        #     - Has phase equilibrium enabled.
        #     - Is not a defined state (solved from balances).
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = True
        tmp_dict["defined_state"] = False            
        self._int_sat_vap_state = self.config.property_package.build_state_block(
            self.flowsheet().time, 
            doc="Thermophysical properties internal saturated vapour state.", 
            **tmp_dict
        )
        self._int_sat_vap_state[:].flow_mol.fix(1)

        return [
            self._int_sat_vap_state,
        ]

    def _add_bounds_to_state_properties(self) -> None:
        """Add lower and/or upper bounds to state properties.

        - Set nonnegativity lower bounds on all inlet/intermediate/outlet flows.
        """
        for sb in (self.inlet_blocks + self.outlet_blocks):
            for t in sb:
                sb[t].flow_mol.setlb(0.0)     

    def _create_references(self) -> None:
        """Create convenient References.

        Creates references to _int_mixed_inlet_state properties:
            - ``bfw_flow_mass``
            - ``bfw_flow_mol``
        """
        self.bfw_flow_mass = Reference(
            self.inlet_water_state[:].flow_mass
        )
        self.bfw_flow_mol = Reference(
            self.inlet_water_state[:].flow_mol
        )
        self.inlet_water_state[:].flow_mol.unfix() 
        self.water = build_package("helmholtz", ["water"], ["Liq"]) 

    def _create_variables(self) -> None:
        """Declare decision/parameter variables for the unit.

        Creates:
            - ``heat_loss`` 
            - ``pressure_loss``
            - ``bfw_temperature``  
            - ``deltaT_superheat``      
        """
        # Get units consistent with the property package
        units_meta = self.config.property_package.get_metadata()

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
            - ``flow_ratio`` 
        """
        # Calculated, always show
        self.flow_ratio = Expression(
            self.flowsheet().time,
            rule=lambda b, t: (
                b.inlet_water_state[t].flow_mol
                /
                (b.inlet_steam_state[t].flow_mol + 1e-9)
            ),
            doc="Ratio of water to steam flows.",
        )

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------
    def _add_material_balances(self) -> None:
        """Material balance equations summary.

        Balances / Constraints:
            - ``overall_material_balance``
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
        
    def _add_energy_balances(self) -> None:
        """Energy balance equations summary.

        Balances / Constraints:
            - ``overall_energy_balance``
            - ``saturated_vap_enthalpy_eq``
        """
        @self.Constraint(
                self.flowsheet().time, 
                doc="Overall energy balance",
                )
        def overall_energy_balance(b, t):
            return (
                sum(
                    i[t].flow_mol * i[t].enth_mol
                    for i in b.inlet_blocks
                )
                ==
                sum(
                    i[t].flow_mol * i[t].enth_mol
                    for i in b.outlet_blocks
                )
                +
                b.heat_loss[t]                
            )
        @self.Constraint(
                self.flowsheet().time, 
                doc="Saturated vapour enthalpy",
                )            
        def saturated_vap_enthalpy_eq(b, t):
            return (
                b.outlet_state[t].enth_mol_sat_phase["Vap"]
                ==
                b._int_sat_vap_state[t].enth_mol
            )           

    def _add_momentum_balances(self) -> None:
        """Momentum balance equations summary.

        Balances / Constraints:
            - ``overall_momentum_balance``
            - ``intlet_water_momentum_balance``    
            - ``saturated_vap_pressure_eq``        
        """
        @self.Constraint(
                self.flowsheet().time, 
                doc="Overall momentum balance",
                )
        def overall_momentum_balance(b, t):
            return (
                b.inlet_steam_state[t].pressure
                == 
                b.outlet_state[t].pressure 
                + 
                b.pressure_loss[t]
            )
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
                b._int_sat_vap_state[t].pressure
                ==
                b.outlet_state[t].pressure
            )                 

    def _add_additional_constraints(self) -> None:
        """Add auxiliary constraints and bounds.

        Constraints: 
            - ``inlet_water_temperature_eq``    
            - ``desuperheating_temperature_eq`` 
        """
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
                doc="Temperature after desuperheating",
                )
        def desuperheating_temperature_eq(b, t):
            return (
                b.outlet_state[t].temperature 
                ==
                b._int_sat_vap_state[t].temperature + b.deltaT_superheat[t]
                )         

    def calculate_scaling_factors(self):
        """Assign scaling factors to improve numerical conditioning.

        Sets scaling factors for performance and auxiliary variables. 
        """
        super().calculate_scaling_factors()
        scaling.set_scaling_factor(self.heat_loss, 1e-3) # kW scale
        scaling.set_scaling_factor(self.pressure_loss, 1e-3) # kPa scale

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
        return {
            "vars": {
                "BFW temperature [K]": self.bfw_temperature[time_point],
                "Degree of superheat target [K]": self.deltaT_superheat[time_point],
                "Heat loss [W]": self.heat_loss[time_point],
                "Pressure loss [Pa]": self.pressure_loss[time_point],
            },
            "exprs": {
                "Water-to-steam flow ratio": self.flow_ratio[time_point],
            },
        }

    def initialize(self, *args, **kwargs):
        """Initialize the Desuperheater unit using :class:`DesuperheaterInitializer`.

        Args:
            *args: Forwarded to ``DesuperheaterInitializer.initialize``.
            **kwargs: Forwarded to ``DesuperheaterInitializer.initialize`` (e.g., solver, options).

        Returns:
            pyomo.opt.results.results_.SolverResults: Results from the initializer's solve.
        """
        init = DesuperheaterInitializer()
        return init.initialize(self, *args, **kwargs)
