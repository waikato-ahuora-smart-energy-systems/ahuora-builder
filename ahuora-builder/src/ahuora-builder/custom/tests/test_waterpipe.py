import pytest
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models.properties import iapws95
from idaes.models_extra.power_generation.unit_models.waterpipe import WaterPipe
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester
from idaes.core.solvers import get_solver

solver = get_solver()

def test_waterpipe_simple():
    '''
    Simple test for the WaterPipe unit model
    '''
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = iapws95.Iapws95ParameterBlock()
    m.fs.pipe = WaterPipe(
        dynamic=False,
        property_package=m.fs.properties,
        has_holdup=True,
        has_heat_transfer=False,
        has_pressure_change=True,
        water_phase="Liq",
        contraction_expansion_at_end="None",
    )
    m.fs.pipe.diameter.fix(0.04)
    m.fs.pipe.length.fix(40)
    m.fs.pipe.number_of_pipes.fix(100)
    m.fs.pipe.elevation_change.fix(25)
    m.fs.pipe.fcorrection_dp.fix(1.0)

    state_args = {"flow_mol": 10000, "pressure": 1.3e7, "enth_mol": 18000}
    initialization_tester(m, dof=3, state_args=state_args, unit=m.fs.pipe)
    m.fs.pipe.inlet.enth_mol.fix()
    m.fs.pipe.inlet.flow_mol.fix()
    m.fs.pipe.inlet.pressure.fix()
    assert degrees_of_freedom(m) == 0

    results = solver.solve(m, tee=True)
    assert pyo.check_optimal_termination(results)
    # Output for debugging
    print("Outlet pressure:", pyo.value(m.fs.pipe.outlet.pressure[0]))
    print("Pressure drop:", pyo.value(m.fs.pipe.deltaP[0]))
    print("Outlet enthalpy:", pyo.value(m.fs.pipe.outlet.enth_mol[0]))