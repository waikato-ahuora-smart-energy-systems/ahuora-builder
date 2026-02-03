from pyomo.environ import Suffix, Var, Expression, Constraint
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.network import Arc, SequentialDecomposition
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
from idaes.core.util.math import smooth_max
from idaes.models.unit_models.separator import SplittingType, EnergySplittingType
from idaes.models.unit_models.mixer import Mixer
from idaes.models.unit_models.heater import Heater
from ahuora_builder.custom.custom_separator import CustomSeparator
from ahuora_builder.custom.simple_separator import SimpleSeparator
from ..inverted import add_inverted, initialise_inverted
__author__ = "Team Ahuora"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("SteamHeader")
class SteamHeaderData(UnitModelBlockData):
    """
    Steam Header unit operation:
    Mixer -> Cooler -> Phase Separator -> Simple Separator
    Separates 100% liquid to condensate_outlet and 100% vapor to splitter outlets.
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
    CONFIG.declare(
        "num_inlets",
        ConfigValue(
            default=2,
            domain=int,
            description="Number of inlets to add" "Index [-1]: Steam makeup",
        ),
    )
    CONFIG.declare(
        "num_outlets",
        ConfigValue(
            default=2,
            domain=int,
            description="Number of outlets to add"
            "Index [-1]: Vent"
            "Index [-2]: Condensate",
        ),
    )

    def build(self):
        super().build()
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        self.inlet_list = [f"inlet_{i+1}" for i in range(self.config.num_inlets)]
        self.outlet_list = [
            f"outlet_{i+1}" for i in range(self.config.num_outlets)
        ] + ["vent"]

        # Create internal units
        self.mixer = Mixer(
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            num_inlets=self.config.num_inlets,
            inlet_list=self.inlet_list,
        )
        self.cooler = Heater(
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            has_pressure_change=True,
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
        )
        self.phase_separator = CustomSeparator(
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            outlet_list=["vapor_outlet", "condensate_outlet"],
            split_basis=SplittingType.phaseFlow,
            energy_split_basis=EnergySplittingType.enthalpy_split,
        )
        self.splitter = SimpleSeparator(
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            outlet_list=self.outlet_list,
        )
        self.unit_ops = [self.mixer, self.cooler, self.phase_separator, self.splitter]

        # Updated internal arcs
        self.mixer_to_cooler_arc = Arc(
            source=self.mixer.outlet, destination=self.cooler.inlet
        )
        self.cooler_to_separator_arc = Arc(
            source=self.cooler.outlet, destination=self.phase_separator.inlet
        )
        self.separator_to_splitter_arc = Arc(
            source=self.phase_separator.vapor_outlet, destination=self.splitter.inlet
        )

        self.inlet_blocks = [getattr(self.mixer, f"{i}_state") for i in self.inlet_list]
        self.outlet_blocks = [getattr(self.splitter, f"{o}_state") for o in self.outlet_list]

        # Declare slack variables for internal use
        self.balance_flow_mol = Var(
            self.flowsheet().time, initialize=0.0, doc="Balance molar flow (negative means makeup is required, positive means venting)", 
            units=pyo.units.mol / pyo.units.s
        )
        
        # This is only used for scaling the smooth_min/smooth_max value.
        self._partial_total_flow_mol = Var(
            self.flowsheet().time, initialize=0.0, doc="Partial total fixed molar flow",
            units=pyo.units.mol / pyo.units.s
        )

        self.makeup_flow_mol = Var(
            self.flowsheet().time, initialize=0.0, doc="Makeup molar flow",
            units=pyo.units.mol / pyo.units.s
        )


        # Add inverted transformers to heat_duty and deltaP
        # (so that positive values correspond to heat loss and pressure drop)

        add_inverted(self.cooler, "heat_duty")
        add_inverted(self.cooler, "deltaP")
        # Declare additional variables and aliases to expose to the user
        self.heat_duty_inverted = Reference(self.cooler.heat_duty_inverted)
        self.deltaP_inverted = Reference(self.cooler.deltaP_inverted)
        self.heat_duty = Reference(self.cooler.heat_duty)
        self.deltaP = Reference(self.cooler.deltaP)
        
        self.total_flow_mol = Reference(
            self.cooler.control_volume.properties_out[:].flow_mol
        )
        self.total_flow_mass = Reference(
            self.cooler.control_volume.properties_out[:].flow_mass
        )
        self.pressure = Reference(
            self.cooler.control_volume.properties_out[:].pressure
            )
        self.temperature = Reference(
            self.cooler.control_volume.properties_out[:].temperature
        )
        self.enth_mol = Reference(
            self.cooler.control_volume.properties_out[:].enth_mol
            )
        self.enth_mass = Reference(
            self.cooler.control_volume.properties_out[:].enth_mass
            )
        self.vapor_frac = Reference(
            self.cooler.control_volume.properties_out[:].vapor_frac
        )
        self.split_flow = self._create_split_flow_references()

        # Condensate liquid always is removed first.
        self.phase_separator.split_fraction[:, "vapor_outlet", "Vap"].fix(1.0)
        self.phase_separator.split_fraction[:, "vapor_outlet", "Liq"].fix(0.0)

        # Additional bounds and constraints
        self._additional_constraints()

        # Expand arcs
        pyo.TransformationFactory("network.expand_arcs").apply_to(self)

        self.cooler_to_separator_arc_expanded.flow_mol_equality.deactivate()

        # add an expression for the degree of superheat

        @self.Expression(self.flowsheet().time)
        def degree_of_superheat(self,t):
            return self.splitter.mixed_state[t].temperature - self.splitter.mixed_state[t].temperature_sat

    def _create_split_flow_references(self):
        self.outlet_idx = pyo.Set(initialize=self.outlet_list)
        # Map each (t, o) to the outlet state's flow var
        ref_map = {}
        for o in self.outlet_list:
            if o != "vent":
                outlet_state_block = getattr(self.splitter, f"{o}_state")
                for t in self.flowsheet().time:
                    ref_map[(t, o)] = outlet_state_block[t].flow_mol

        return Reference(ref_map)

    def _additional_constraints(self):
        """
        Additional constraints.
        """

        """
        1) Set lower bounds on flow variables for all external ports.
        """
        [
            state_block[t].flow_mol.setlb(0.0)
            for state_block in (self.inlet_blocks + self.outlet_blocks)
            for t in state_block
        ]

        """
        2) Add overall material balance equation.
        """

        # Write phase-component balances
        @self.Constraint(self.flowsheet().time, doc="Material balance equation")
        def material_balance_equation(b, t):
            pc_set = b.outlet_blocks[0].phase_component_set
            comp_ls = b.outlet_blocks[0].component_list
            phase_ls = b.outlet_blocks[0].phase_list
            return (
                0
                == sum(
                    sum(
                        b.mixer.mixed_state[t].get_material_flow_terms(p, j)
                        - (b.phase_separator.mixed_state[t].get_material_flow_terms(p, j)
                        - b.splitter.vent_state[t].get_material_flow_terms(p, j))
                        for j in comp_ls
                        if (p, j) in pc_set
                    )
                    for p in phase_ls
                )
                - b.balance_flow_mol[t]
            )

        """
        3) Add vent flow balance (scaled by the total known flows).
        """

        @self.Constraint(
            self.flowsheet().time,
            doc="Material balance of known inlet and outlet flows. "
            "Only used for scaling the steam vent min flow, doesn't actually matter.",
        )
        def partial_material_balance_equation(b, t):
            pc_set = b.inlet_blocks[0].phase_component_set
            comp_ls = b.inlet_blocks[0].component_list
            phase_ls = b.inlet_blocks[0].phase_list
            return (
                0
                == sum(
                    sum(
                        sum(
                            o[t].get_material_flow_terms(p, j)
                            for o in b.inlet_blocks[:-1] + b.outlet_blocks[:-2]
                        )
                        for j in comp_ls
                        if (p, j) in pc_set
                    )
                    for p in phase_ls
                )
                - b._partial_total_flow_mol[t]
            )

        eps = 1e-5  # smoothing parameter; smaller = closer to exact max, larger = smoother
        # calculate amount for vent flow: positive balance amount
        @self.Constraint(self.flowsheet().time, doc="Steam vent flow balance.")
        def vent_flow_balance(b, t):
            return 0 == (
                smooth_max(
                    b.balance_flow_mol[t] / (b._partial_total_flow_mol[t] + 1e-6),
                    0.0,
                    eps,
                )
                * (b._partial_total_flow_mol[t] + 1e-6)
                - b.splitter.vent_state[t].flow_mol
            )
        
        # if balance is negative, vent will be zero. makeup will be required.
        @self.Constraint(self.flowsheet().time, doc="Steam makeup balance")
        def makeup_flow_balance(b, t):
            return b.makeup_flow_mol[t] == (
                b.splitter.vent_state[t].flow_mol - b.balance_flow_mol[t]
            )


    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        [getattr(o, "calculate_scaling_factors")() for o in self.unit_ops]

    def initialize_build(self, outlvl=idaeslog.NOTSET, **kwargs):
        """
        Initialize the Steam Header unit using Sequential Decomposition to determine optimal order
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")

        init_log.info(
            f"Starting {self.parent_block().__class__.__name__} initialization using Sequential Decomposition"
        )
        # Initialize the inverted values
        initialise_inverted(self.cooler, "heat_duty")
        initialise_inverted(self.cooler, "deltaP")

        # create Sequential Decomposition object
        seq = SequentialDecomposition()
        seq.options.select_tear_method = "heuristic"
        seq.options.tear_method = "Wegstein"
        seq.options.iterLim = 1

        # create computation graph
        G = seq.create_graph(self)
        heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
        # get calculation order
        order = seq.calculation_order(G)

        for o in heuristic_tear_set:
            print(o.name)

        print("Initialization order:")
        for o in order:
            print(o[0].name)

        # define unit initialisation function
        def init_unit(unit):
            unit.initialize(outlvl=outlvl, **kwargs)

        # run sequential decomposition
        seq.run(self, init_unit)

    def _get_stream_table_contents(self, time_point=0):
        """
        Create stream table showing all inlets, outlets, and liquid outlet
        """
        io_dict = {}

        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name)

        for outlet_name in self.outlet_list:
            io_dict[outlet_name] = getattr(self, outlet_name)

        io_dict["condensate_outlet"] = self.condensate_outlet

        return create_stream_table_dataframe(io_dict, time_point=time_point)
