### Imports
from pyomo.environ import ConcreteModel, SolverFactory, SolverStatus, TerminationCondition, Block, TransformationFactory, assert_optimal_termination
from pyomo.network import SequentialDecomposition, Port, Arc
from pyomo.core.base.units_container import _PyomoUnit, units 
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom
from idaes.core.util.tables import _get_state_from_port
import idaes.logger as idaeslog
from property_packages.build_package import build_package
from idaes.models.unit_models.pressure_changer import Pump
from idaes.models.unit_models.heater import Heater
from ..energy_mixer import EnergyMixer
from ..power_property_package import PowerParameterBlock
from pyomo.environ import value


def test_energy_mixer():
    m = ConcreteModel()
    m.fs = FlowsheetBlock()
    m.fs.power_property_package = PowerParameterBlock()

    m.fs.mixer = EnergyMixer(
        property_package=m.fs.power_property_package,
        num_inlets=3,
        )
        
    
    

    m.fs.mixer.inlet_1.power.fix(100 * units.kW) 
    m.fs.mixer.inlet_2.power.fix(1000 * units.W)
    m.fs.mixer.inlet_3.power.fix(10 * units.kW)
    m.fs.mixer.efficiency.fix(0.90)

    # Check that the model is fully specified
    assert degrees_of_freedom(m) == 0 

    # check that it solves correctly

    solver = SolverFactory("ipopt")
    solver.solve(m)
    #assert m.fs.mixer.outlet.power[0].value == 111 * units.kW
    assert abs(m.fs.mixer.outlet.power[0].value - 999e2) < 1e-3
    


