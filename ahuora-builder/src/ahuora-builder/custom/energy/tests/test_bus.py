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
from ..power_property_package import PowerParameterBlock
from ..bus import Bus
from pyomo.environ import value

def test_bus():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.power_property_package = PowerParameterBlock()

    m.fs.bus = Bus(
        property_package=m.fs.power_property_package,
        num_inlets=2,
        
        
        )
    m.fs.bus.inlet_1.power.fix(100 * units.kW) 
    m.fs.bus.inlet_2.power.fix(50 * units.kW)
    

    m.fs.bus.eq_power_balance.pprint()

    #m.fs.bus.display()
    
    # Check that the model is fully specified
    assert degrees_of_freedom(m.fs.bus) == 0 

    # check that it solves correctly

    solver = SolverFactory("ipopt")
    solver.solve(m)
    print(*"Post solve power:")

    assert value(m.fs.bus.properties_out[0].power)- 150e3 == 0