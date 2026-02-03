from idaes.core import declare_process_block_class, MaterialBalanceType
from .water_tank_with_units import WaterTankData
from .add_initial_dynamics import add_initial_dynamics
from pyomo.environ import Reference, Var
from pyomo.dae import DerivativeVar
from pyomo.environ import units as pyunit
from .inverted import add_inverted, initialise_inverted


def CustomTank(*args, **kwargs):
    # In dynamics mode, we need to use the default material balance type of componentPhase.
    # This does balances for liquid and vapor phases separately. This is needed becasue 
    # accumulation terms are phase specific.
    # However, in steady-state, this is not necessary, because we don't have accumulation terms.
    # So this would make the system over-defined, as the state block defines the phase equilibrium already.
    is_dynamic = kwargs.get("dynamic")
    if is_dynamic:
        kwargs["material_balance_type"] = MaterialBalanceType.componentPhase
    else:
        kwargs["material_balance_type"] = MaterialBalanceType.componentTotal
    
    return DynamicTank(*args, **kwargs)
        

@declare_process_block_class("DynamicTank")
class DynamicTankData(WaterTankData):
    """
    Water tank model with dynamic capabilities.
    Some extra properties are added to IDAES's tank model to allow for easier specification of initial conditions.
    """

    def build(self, *args, **kwargs):
        """
        Build method for the DynamicHeaterData class.
        This method initializes the control volume and sets up the model.
        """

        super().build(*args, **kwargs)

        self.flow_rate_out = Reference(self.outlet.flow_mol)

        # add deltaP_inverted as a property
        add_inverted(self,"heat_duty")

        # Because it's hard to specify the initial conditions directly,
        # Create a state block for the initial conditions.

        if not self.config.dynamic:
            return # There is no need to add these extra properties.        

        self.initial_block = self.config.property_package.build_state_block(
            [0],
            defined_state=True,
        )

        if len(self.config.property_package.component_list) > 1:

            # We are assuming that the composition does not change at the initial time step. In theory it could, but
            # we can worry about that later.
            @self.Constraint(
                self.config.property_package.component_list,
                doc="Initial composition constraint",
            )
            def initial_composition_constraint(b, j):
                return (
                    b.initial_block[0].mole_frac_comp[j]
                    == b.control_volume.properties_in[0].mole_frac_comp[j]
                )

        # The initial temperature, pressure, and flow amount is set by the user.
        self.initial_pressure = Var(initialize=101325, units=pyunit.Pa)
        @self.Constraint(doc="Initial pressure constraint")
        def initial_pressure_constraint(b):
            return b.initial_block[0].pressure == b.initial_pressure

        self.initial_holdup = Var(initialize=300, units=pyunit.mol)
        @self.Constraint(doc="Initial flow constraint")
        def initial_holdup_mol_constraint(b):
            return (
                b.initial_block[0].flow_mol * pyunit.s == b.initial_holdup
            )  # cancel out the seconds as we are using it for holdup not accumulation.
        

        self.initial_level = Var(initialize=300, units=pyunit.m)
        @self.Constraint(doc="Initial level constraint")
        def initial_level_constraint(b):
            return b.initial_holdup == b.tank_cross_sect_area * b.initial_level * b.initial_block[0].dens_mol

        self.initial_temperature = Var(initialize=300, units=pyunit.K)
        @self.Constraint(doc="Initial temperature constraint")
        def initial_temperature_constraint(b):
            return b.initial_block[0].temperature == b.initial_temperature

        # The temperature, pressure and flow are used to calculate the other properties.
        @self.Constraint(
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Defining accumulation",
        )
        def initial_material_conditions_constraint(b, p, j):
            return (
                b.initial_block[0].flow_mol
                * b.initial_block[0].mole_frac_phase_comp[p, j]
                + b.control_volume.material_accumulation[0, p, j]
                == b.control_volume.material_holdup[0, p, j]
            )

        @self.Constraint(
            self.config.property_package.phase_list, doc="Defining accumulation"
        )
        def initial_energy_conditions_constraint(b, p):
            return (
                b.initial_block[0].flow_mol
                * b.initial_block[0].phase_frac[p]
                * b.initial_block[0].enth_mol_phase[p]
                + b.control_volume.energy_accumulation[0, p]
                == b.control_volume.energy_holdup[0, p]
            )
        
        

    def initialize(self, *args, **kwargs):
        """
        Initialization method for the DynamicTankData class.
        This method initializes the control volume and sets up the model.
        """
        # Copy initial conditions from inverted properties
        initialise_inverted(self,"heat_duty")

        if self.config.dynamic:
            self.initial_block.initialize()
        
        super().initialize(*args, **kwargs)
