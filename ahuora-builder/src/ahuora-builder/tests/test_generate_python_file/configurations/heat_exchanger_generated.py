### Imports
from pyomo.environ import ConcreteModel, SolverFactory, SolverStatus, TerminationCondition, Block, TransformationFactory, assert_optimal_termination
from pyomo.network import SequentialDecomposition, Port, Arc
from pyomo.core.base.units_container import _PyomoUnit, units as pyomo_units
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom
from idaes.core.util.tables import _get_state_from_port
import idaes.logger as idaeslog
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from property_packages.build_package import build_package
from ahuora_builder.custom.custom_heat_exchanger import CustomHeatExchanger


### Utility Methods
def units(item: str) -> _PyomoUnit:
    ureg = pyomo_units._pint_registry
    pint_unit = getattr(ureg, item)
    return _PyomoUnit(pint_unit, ureg)


### Build Model
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)

# Set up property packages
m.fs.PP__1 = build_package(
    "helmholtz",
    ["h2o"],
)

# Create unit models

# heat_exchanger-1
m.fs.heat_exchanger_1 = CustomHeatExchanger(
    delta_temperature_callback=delta_temperature_underwood_callback,
    hot_side={"property_package": m.fs.PP__1,"has_pressure_change": True},
    cold_side={"property_package": m.fs.PP__1,"has_pressure_change": True},
    dynamic=False
)
m.fs.heat_exchanger_1.overall_heat_transfer_coefficient.fix(100.0 * units("W/(m**2*K)"))
m.fs.heat_exchanger_1.area.fix(1000.0 * units("m**2"))
m.fs.heat_exchanger_1.hot_side.deltaP.fix(100.0 * units("Pa"))
m.fs.heat_exchanger_1.cold_side.deltaP.fix(100.0 * units("Pa"))
sb = _get_state_from_port(m.fs.heat_exchanger_1.hot_side_inlet, 0)
sb.constrain_component(sb.pressure, 101325.0 * units("Pa"))
sb.constrain_component(sb.enth_mol, 40000.0 * units("J/mol"))
sb.constrain_component(sb.flow_mol, 100.0 * units("mol/s"))
sb = _get_state_from_port(m.fs.heat_exchanger_1.cold_side_inlet, 0)
sb.constrain_component(sb.pressure, 101325.0 * units("Pa"))
sb.constrain_component(sb.enth_mol, 30000.0 * units("J/mol"))
sb.constrain_component(sb.flow_mol, 100.0 * units("mol/s"))


### Check Model Status
report_statistics(m)
print("Degrees of freedom:", degrees_of_freedom(m))


### Initialize Model
m.fs.heat_exchanger_1.initialize(outlvl=idaeslog.INFO)


### Solve
opt = SolverFactory("ipopt")
res = opt.solve(m, tee=True)
assert_optimal_termination(res)


### Report
m.fs.heat_exchanger_1.report()
