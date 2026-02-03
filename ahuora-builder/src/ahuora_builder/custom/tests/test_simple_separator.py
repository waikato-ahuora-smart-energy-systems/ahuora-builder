import pytest
from pyomo.environ import ConcreteModel, SolverFactory, value
from idaes.core import FlowsheetBlock
from idaes.models.properties.iapws95 import Iapws95ParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from ahuora_builder.custom.simple_separator import SimpleSeparator


def test_simple_separator_iapws95():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Property package: IAPWS95 (pure water/steam)
    m.fs.properties = Iapws95ParameterBlock()

    # Add SimpleSeparator with 2 outlets
    m.fs.sep = SimpleSeparator(
        property_package=m.fs.properties,
        num_outlets=2,
    )

    # Fix inlet state
    m.fs.sep.inlet.flow_mol[0].fix(100.0)      # mol/s
    m.fs.sep.inlet.enth_mol[0].fix(50000.0)    # J/mol (example enthalpy)
    m.fs.sep.inlet.pressure[0].fix(101325.0)   # Pa

    # Fix split fractions
    m.fs.sep.outlet_1.flow_mol[0].fix(70.0)  # mol/s

    # Model should be fully specified
    assert degrees_of_freedom(m) == 0

    # Initialize and solve
    m.fs.sep.initialize()
    solver = SolverFactory("ipopt")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == "optimal"

    # Check material balance
    inlet_flow = value(m.fs.sep.inlet.flow_mol[0])
    out1_flow = value(m.fs.sep.outlet_1.flow_mol[0])
    out2_flow = value(m.fs.sep.outlet_2.flow_mol[0])

    assert abs(inlet_flow - (out1_flow + out2_flow)) <= 1e-6
    assert abs(out1_flow - 0.7 * inlet_flow) <= 1e-6
    assert abs(out2_flow - 0.3 * inlet_flow) <= 1e-6
