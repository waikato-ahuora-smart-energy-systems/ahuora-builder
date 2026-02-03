### Imports
from pyomo.environ import ConcreteModel, SolverFactory, SolverStatus, TerminationCondition, Block, TransformationFactory, assert_optimal_termination
from pyomo.network import SequentialDecomposition, Port, Arc
from pyomo.core.base.units_container import _PyomoUnit, units as pyomo_units
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom
from idaes.core.util.tables import _get_state_from_port
import idaes.logger as idaeslog
from property_packages.build_package import build_package
from ahuora_builder.custom.custom_pump import CustomPump
from ahuora_builder.custom.custom_heater import DynamicHeater


### Utility Methods
def units(item: str) -> _PyomoUnit:
    ureg = pyomo_units._pint_registry
    pint_unit = getattr(ureg, item)
    return _PyomoUnit(pint_unit, ureg)

def init_unit(unit: Block) -> None:
    unit.initialize(outlvl=idaeslog.INFO)


### Build Model
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)

# Set up property packages
m.fs.PP__1 = build_package(
    "helmholtz",
    ["h2o"],
)

# Create unit models

# Pump-1
m.fs.Pump_1 = CustomPump(
    property_package=m.fs.PP__1,
    dynamic=False,
    power_property_package=m.fs.power_property_package
)
m.fs.Pump_1.efficiency_pump.fix(0.8 * units("dimensionless"))
sb = _get_state_from_port(m.fs.Pump_1.inlet, 0)
sb.constrain_component(sb.pressure, 11000000.0 * units("Pa"))
sb.constrain_component(sb.enth_mol, 125.0 * units("J/mol"))
sb.constrain_component(sb.flow_mol, 431.0 * units("mol/s"))
sb = _get_state_from_port(m.fs.Pump_1.outlet, 0)
sb.constrain_component(sb.pressure, 22000000.0 * units("Pa"))

# Heater-1
m.fs.Heater_1 = DynamicHeater(
    property_package=m.fs.PP__1,
    has_pressure_change=None,
    dynamic=None,
    has_holdup=None
)
m.fs.Heater_1.heat_duty.fix(100.0 * units("J/s"))


### Connect Unit Models
m.fs.arc_1 = Arc(source=m.fs.Pump_1.outlet, destination=m.fs.Heater_1.inlet)


### Check Model Status
report_statistics(m)
print("Degrees of freedom:", degrees_of_freedom(m))


### Initialize Model
TransformationFactory("network.expand_arcs").apply_to(m)
seq = SequentialDecomposition()
seq.set_tear_set([])
seq.run(m, init_unit)


### Solve
opt = SolverFactory("ipopt")
res = opt.solve(m, tee=True)
assert_optimal_termination(res)


### Report
m.fs.Pump_1.report()
m.fs.Heater_1.report()
