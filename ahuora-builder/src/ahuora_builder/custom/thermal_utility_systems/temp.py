# TODO: 
# backend/idaes_service/solver/custom/steam_header.py [DONE]
# backend/idaes_service/solver/custom/tests/test_steam_header.py [DONE]
# backend/idaes_service/solver/methods/adapter_library.py
# backend/idaes_factory/adapters/unit_models/header_adapter.py
# backend/idaes_factory/adapters/adapter_library.py
# backend/core/auxiliary/enums/unitOpData.py
# backend/flowsheetInternals/unitops/config/config_base.py
# backend/flowsheetInternals/unitops/config/objects/header_config.py
# frontend/src/data/objects.json

from pyomo.environ import Suffix, Var
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.network import Arc
from pyomo.core.base.reference import Reference
import pyomo.environ as pyo

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.config import is_physical_parameter_block
import idaes.logger as idaeslog
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.models.unit_models.mixer import Mixer
from idaes.models.unit_models.heater import Heater

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("Desuperheater")
class DesuperheaterData(UnitModelBlockData):
    """
    Desuperheater unit operation:
    Superheated steam + water -> Mixer -> Losses -> Saturated Vapour
    Desuperheater aims to remove steam superhear by direct contact with boiler feed water.
    Uses Sequential Decomposition for optimal initialization order.
    """

    CONFIG = UnitModelBlockData.CONFIG()
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
        ),
    )


    def build(self):
        super().build()
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Create internal units
        self.mixer = Mixer(
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            num_inlets=2
        )
        self.losses = Heater(
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            has_pressure_change=True,
        )

        # Build saturated vapour state block
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = True
        tmp_dict["defined_state"] = False

        self._properties_sat = self.config.property_package.build_state_block(
            self.flowsheet().time, 
            doc="saturated vapour properties at outlet", 
            **tmp_dict
        )

        self.unit_ops = [self.mixer, self.losses, self._properties_sat]

        # Create internal arcs
        self.condenser_to_subcooler = Arc(
            source=self.mixer.outlet,
            destination=self.losses.inlet,
        )

        # Expand arcs
        pyo.TransformationFactory("network.expand_arcs").apply_to(self)

        # Identify the exposed ports
        inlet_exposed_ls = [
            (self.mixer, "steam_inlet", "inlet_1"),
            (self.mixer, "water_inlet", "inlet_2"),
        ]
        outlet_exposed_ls = [
            (self.losses, "steam_outlet", "outlet"),
        ]

        self.inlet_list, self.inlet_blocks = self._construct_exposed_ports(inlet_exposed_ls)
        self.outlet_list, self.outlet_blocks = self._construct_exposed_ports(outlet_exposed_ls)
        
        # Declare additional variables and provide alias of existing ones to expose to the user        
        self.deltaT_supheat = Var(
            self.flowsheet().time, 
            domain=pyo.NonNegativeReals,
            # initialize=0.0, 
            units=pyo.units.K, 
            doc="The degree of steam superheat remaining."
        )
        self.water_temperature = Reference(
            self.water_inlet_state[:].temperature
        )        
        self.water_flow_mol = Reference(
            self.water_inlet_state[:].flow_mol
        )
        self.water_flow_mass = Reference(
            self.water_inlet_state[:].flow_mass
        )
        self.heat_duty = Reference(
            self.losses.heat_duty[:]
        )
        self.deltaP = Reference(
            self.losses.deltaP[:]
        ) 

        # Declare internal variables and references for internal use
        self._T_sat = Reference(
            self._properties_sat[:].temperature
        )

        # Additional bounds and constraints
        self._additional_constraints()
               
    def _additional_constraints(self):
        """
        Additional constraints.
        """
        
        """
        1. Pressure equality for saturation state block
        """
        @self.Constraint(
            self.flowsheet().time, doc="Pressure equality for saturation state block"
        )
        def outlet_pressure_eql(b, t):
            return (
                b._properties_sat[t].pressure == b.losses.control_volume.properties_out[t].pressure
            )

        """
        2. Vapour quality for saturation state block
        """
        @self.Constraint(
            self.flowsheet().time, doc="Pressure equality for saturation state block"
        )
        def vapour_quality_eq(b, t):
            return (
                b._properties_sat[t].vapor_frac == 0.5
            )
        
        """
        3. Mixer pressure inlet equality for steam and water 
        """
        @self.Constraint(
                self.flowsheet().time, doc="Mixer pressure inlet equality for steam and water"
        )
        def mixer_inlet_pressure_eql(b, t):
            return (
                b.steam_inlet_state[t].pressure == b.water_inlet_state[t].pressure
            )
                
        """
        4. Superheat temperature balance
        """
        @self.Constraint(self.flowsheet().time, doc="Temperature superheat balance")
        def superheat_temperature_eqn(b, t):
            return (
                b.deltaT_supheat[t] + 0.001 == b.steam_outlet_state[t].temperature - b._T_sat[t]
            )
                
    def _construct_exposed_ports(self, exposed_ls):
        """
        exposed_ls: list of [(unit, public_name, internal_basename)]
        """
        names, states = [], []

        for unit, public_name, internal_basename in exposed_ls:
            # 1) Locate the state block on the child unit
            # Try <basename>_state (Mixer & many IDAES units)
            state = getattr(unit, f"{internal_basename}_state", None)

            # Fall back to control_volume properties for CV-based units (e.g., Heater)
            if state is None and hasattr(unit, "control_volume"):
                cv = unit.control_volume
                if internal_basename.startswith("inlet") and hasattr(cv, "properties_in"):
                    state = cv.properties_in
                elif internal_basename.startswith("outlet") and hasattr(cv, "properties_out"):
                    state = cv.properties_out

            if state is None:
                raise AttributeError(
                    f"{unit.name} has no state for '{internal_basename}'. "
                    f"Tried '{internal_basename}_state' and control_volume properties."
                )

            # 2) Expose the state on this wrapper
            #    (a) keep a convenient Reference to the time-indexed state block
            setattr(self, f"{public_name}_state", Reference(state[:]))

            #    (b) create a public Port mapped to that state
            #        NOTE: UnitModelBlock.add_port builds a Port with the vars
            #        defined by the property package’s port members.
            self.add_port(name=public_name, block=state)

            names.append(public_name)
            states.append(state)

        return names, states

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        for o in self.unit_ops:
            if hasattr(o, "calculate_scaling_factors"):
                o.calculate_scaling_factors()
                
    def _get_stream_table_contents(self, time_point=0):
        """
        Create stream table showing all inlets and outlets
        """
        io_dict = {}
        
        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name)
        
        for outlet_name in self.outlet_list:
            io_dict[outlet_name] = getattr(self, outlet_name)
            
        return create_stream_table_dataframe(io_dict, time_point=time_point)
     
    def _copy_port_state(self, src_port, snk_port, t):
        for name in ("flow_mol", "pressure", "enth_mol", "temperature", "enth_mass", "vapor_frac"):
            if hasattr(src_port, name) and hasattr(snk_port, name):
                src = getattr(src_port, name)
                snk = getattr(snk_port, name)
                if (t in src) and (t in snk) and pyo.is_variable_type(snk[t]):
                    try:
                        snk[t].set_value(pyo.value(src[t]))
                    except Exception:
                        pass  

    def _hold_state_vars(self, sb):
        held = {}
        if hasattr(sb, "define_state_vars"):
            for k, v in sb.define_state_vars().items():
                if pyo.is_variable_type(v) and not v.fixed:
                    v.fix()
                    held[k] = False
                else:
                    held[k] = True
        return held

    def _release_state_vars(self, sb, held):
        if hasattr(sb, "define_state_vars"):
            for k, was_fixed in held.items():
                v = sb.define_state_vars()[k]
                if was_fixed is False and pyo.is_variable_type(v):
                    v.unfix()

    def initialize_build(self, outlvl=idaeslog.NOTSET, **kwargs):
        """
        Initialize the Desuperheater without using state_args for sub-units.

        Steps:
        1) Initialize Mixer (no state_args; it reads from connected inlet ports).
        2) Initialize Heater with temporary ΔP=0 and Q=0, then restore.
        3) Seed and (if supported) initialize saturated-vapor state block.
        4) Seed deltaT_supheat.
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        t0 = list(self.flowsheet().time)[0]

        # 1) Mixer
        init_log.info("Initializing Mixer (no state_args)...")
        try:
            # Inlet states should already be provided via self.steam_inlet / self.water_inlet
            # and are wired to mixer.inlet_1 / mixer.inlet_2 via the exposed ports.
            self.water_flow_mol[t0].fix(0.0)
            self.water_flow_mol[t0].unfix()
            self.mixer.inlet_2.pressure[t0].fix(
                pyo.value(
                    self.mixer.inlet_1.pressure[t0]
                )
            )
            self.mixer.inlet_2.pressure[t0].unfix()
            self.mixer.initialize(outlvl=outlvl)
        except Exception as err:
            init_log.error(f"Mixer initialization failed: {err}")
            raise

        # 2) Heater (losses)
        init_log.info("Initializing Heater (losses) with temporary ΔP=0 and Q=0...")
        self._copy_port_state(
            src_port=self.mixer.outlet,
            snk_port=self.losses.inlet,
            t=t0
        )
        held = self._hold_state_vars(self.losses.control_volume.properties_in[t0])
        try:
            self.losses.initialize(outlvl=outlvl)
        finally:
            self._release_state_vars(self.losses.control_volume.properties_in[t0], held)

        # 3) Saturated-vapor state block seeding
        init_log.info("Seeding saturated-vapor state block...")
        try:
            # Set seed values directly
            self._properties_sat[t0].pressure.set_value(
                pyo.value(self.losses.outlet.pressure[t0])
            )
            self._properties_sat[t0].enth_mol.set_value(
                pyo.value(self.losses.outlet.enth_mol[t0])
            )

            # If the block exposes its own initialize(outlvl=...), call it
            if hasattr(self._properties_sat[t0], "initialize"):
                # Many property packages implement initialize on the *indexed* block,
                # i.e. call on the parent indexed component:
                try:
                    self._properties_sat.initialize(outlvl=outlvl)
                except TypeError:
                    # Some implementations only support per-timepoint initialize
                    self._properties_sat[t0].initialize(outlvl=outlvl)

        except Exception as err:
            init_log.error(f"Saturation state initialization failed: {err}")
            raise

        init_log.info("Desuperheater initialization complete.")
