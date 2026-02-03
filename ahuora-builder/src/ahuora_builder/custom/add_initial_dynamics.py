from pyomo.environ import Reference, Var, units
from pyomo.dae import DerivativeVar


def add_initial_dynamics(unit_model):
    """
    Adds the reference variables for initial holdup and initial accumulation. 
    Makes it easier for us to set initial conditions in the frontend, as we can reference them directly.
    """
    if unit_model.config.dynamic:
            # add initial holdup reference
            unit_model.initial_material_holdup = Reference(unit_model.control_volume.material_holdup[0,:,:])
            unit_model.initial_energy_holdup = Reference(unit_model.control_volume.energy_holdup[0,:])

            # For some reason we can't do references to the initialaccumulation variables,
            # Error ( Can only take the derivative of a Varcomponent.)
            #  so we create them as vars
            unit_model.initial_material_accumulation = Var(unit_model.config.property_package.phase_list, unit_model.config.property_package.component_list, initialize=0,units=units.mol/units.s)
            unit_model.initial_energy_accumulation = Var(unit_model.config.property_package.phase_list, initialize=0,units=units.kW)


            @unit_model.Constraint(
                unit_model.config.property_package.phase_list,
                unit_model.config.property_package.component_list,
                doc="Initial material accumulation constraint"
            )
            def initial_material_accumulation_constraint(b, p, j):
                return b.initial_material_accumulation[p, j] == b.control_volume.material_accumulation[0, p, j]
            
            @unit_model.Constraint(
                unit_model.config.property_package.phase_list,
                doc="Initial energy accumulation constraint"
            )
            def initial_energy_accumulation_constraint(b, p):
                return b.initial_energy_accumulation[p] == b.control_volume.energy_accumulation[0, p]