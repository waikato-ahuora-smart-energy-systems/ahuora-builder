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
from ..mainDistributionBoard import MDB
from pyomo.environ import value

def test_mdb():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.power_property_package = PowerParameterBlock()
    m.fs.mdb = MDB(
        property_package=m.fs.power_property_package,
        num_inlets=1,
        num_outlets=3,
        )
    m.fs.mdb.inlet_1.power.fix(10 * units.kW)
    m.fs.mdb.priorities[0,'outlet_1'].fix(5* units.kW)
    m.fs.mdb.priorities[0,'outlet_2'].fix(3* units.kW)

    
    # Check that the model is fully specified
    assert degrees_of_freedom(m.fs.mdb) == 0 

    # check that it solves correctly
    solver = SolverFactory("ipopt")
    solver.solve(m)
    assert abs(value(m.fs.mdb.outlet_3.power[0])) <= 2.0005e3 
    
    #Reduce input power and check that it doesn't give power to the last outlet:
    m.fs.mdb.inlet_1.power.fix(8 * units.kW)
    solver = SolverFactory("ipopt")
    solver.solve(m)
    assert abs(value(m.fs.mdb.outlet_3.power[0])) <= 0.0005

    #Increase input power and check that it gives all excess power to the last outlet:
    m.fs.mdb.inlet_1.power.fix(18 * units.kW)
    solver = SolverFactory("ipopt")
    solver.solve(m)
    assert abs(value(m.fs.mdb.outlet_3.power[0])) <= 10.0005e3