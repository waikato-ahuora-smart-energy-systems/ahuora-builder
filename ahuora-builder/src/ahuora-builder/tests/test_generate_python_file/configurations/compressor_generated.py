### Imports
from pyomo.environ import ConcreteModel, SolverFactory, SolverStatus, TerminationCondition, Block, TransformationFactory, assert_optimal_termination
from pyomo.network import SequentialDecomposition, Port, Arc
from pyomo.core.base.units_container import _PyomoUnit, units as pyomo_units
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom
from idaes.core.util.tables import _get_state_from_port
import idaes.logger as idaeslog
from property_packages.build_package import build_package
from ahuora_builder.custom.custom_compressor import CustomCompressor


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

# Compressor-1
m.fs.Compressor_1 = CustomCompressor(
    property_package=m.fs.PP__1,
    power_property_package=m.fs.power_property_package,
    dynamic=False
)
m.fs.Compressor_1.deltaP.fix(50000.0 * units("Pa"))
m.fs.Compressor_1.efficiency_isentropic.fix(0.9 * units("None"))
sb = _get_state_from_port(m.fs.Compressor_1.inlet, 0)
sb.constrain_component(sb.pressure, 101325.0 * units("Pa"))
sb.constrain_component(sb.enth_mol, 4000.0 * units("J/mol"))
sb.constrain_component(sb.flow_mol, 100.0 * units("mol/s"))
sb.constrain_component(sb.mole_frac_comp[('h2o',)], 1.0 * units("None"))


### Check Model Status
report_statistics(m)
print("Degrees of freedom:", degrees_of_freedom(m))


### Initialize Model
m.fs.Compressor_1.initialize(outlvl=idaeslog.INFO)


### Solve
opt = SolverFactory("ipopt")
res = opt.solve(m, tee=True)
assert_optimal_termination(res)


### Report
m.fs.Compressor_1.report()
