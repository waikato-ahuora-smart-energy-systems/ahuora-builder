import pytest

from ahuora_builder.custom.thermal_utility_systems.simple_heat_pump import (
    SimpleHeatPump,
)

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
#from property_packages.build_package import build_package # build package still broken for windows users
from idaes.models.properties.general_helmholtz import (
    HelmholtzParameterBlock,
    PhaseType,
    StateVars,
    AmountBasis,
    )

from pyomo.network import Arc, SequentialDecomposition
import idaes.logger as idaeslog
import math

def _make_case():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic = False)
    m.fs.water = HelmholtzParameterBlock(
                    pure_component="h2o",
                    phase_presentation=PhaseType.LG,
                    state_vars=StateVars.PH,
                    amount_basis=AmountBasis.MOLE,
                    )
    #build_package("helmholtz", ["water"], ["Liq", "Vap"])

    m.fs.hp = SimpleHeatPump(source={
        "property_package": m.fs.water,
        "has_pressure_change": False,
    }, sink={
        "property_package": m.fs.water,
        "has_pressure_change": False,
    })
    
    # Set source (hot side) conditions
    T_source_in = (80 + 273.15) * pyo.units.K  # K
    P_source_in = 4e5 * pyo.units.Pa  # Pa
    F_source_in = 29 * 15.4 * pyo.units.mol / pyo.units.s  # mol/s

    m.fs.hp.source_inlet.flow_mol.fix(F_source_in)  # mol/s
    m.fs.hp.source_inlet.pressure.fix(P_source_in)  # 4 Bar
    m.fs.hp.source_inlet.enth_mol.fix(
        m.fs.water.htpx(T_source_in , m.fs.hp.source_inlet.pressure[0])
    )  # J/mol
    m.fs.hp.approach_temperature.fix(10)  # K

    # Set sink (cold side) conditions
    T_sink_in = (90 + 273.15) * pyo.units.K  # K
    P_sink_in = 8e5 * pyo.units.Pa  # Pa
    F_sink_in = 1.1 * 15.4 * pyo.units.mol / pyo.units.s  # mol/s

    m.fs.hp.sink_inlet.flow_mol.fix(F_sink_in)  # mol/s
    m.fs.hp.sink_inlet.pressure.fix(P_sink_in)  # 10 Bar
    m.fs.hp.sink_inlet.enth_mol.fix(
        m.fs.water.htpx(T_sink_in, m.fs.hp.sink_inlet.pressure[0])
    )  # J/mol
    
    return m
 
def _assert_solution(m):
    eps_rel = 1e-4 
    eps_abs = 0.1
    m.fs.hp.initialize()

    assert degrees_of_freedom(m) == 0

    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=True)
    
    # Check heat balances Qsink = Qsource + W
    Q_sink = pyo.value(m.fs.hp.sink_inlet.flow_mol[0] * (m.fs.hp.sink_outlet.enth_mol[0] - m.fs.hp.sink_inlet.enth_mol[0] ))
    Q_source = pyo.value(m.fs.hp.source_inlet.flow_mol[0] * (m.fs.hp.source_inlet.enth_mol[0] - m.fs.hp.source_outlet.enth_mol[0] ))
    
    assert Q_sink - pyo.value(m.fs.hp.work_mechanical[0]) - Q_source == pytest.approx(0, rel=eps_rel, abs=eps_abs)     
    assert abs(pyo.value(m.fs.hp.source.heat[0])) == pytest.approx(Q_source, rel=eps_rel, abs=eps_abs)
    assert abs(pyo.value(m.fs.hp.sink.heat[0])) == pytest.approx(Q_sink, rel=eps_rel, abs=eps_abs)
    
    # Check that CoP is calculated correctly
    assert pyo.value(m.fs.hp.coefficient_of_performance) == pytest.approx(Q_sink / pyo.value(m.fs.hp.work_mechanical[0]), rel=eps_rel, abs=eps_abs)
    
    # Check that duties and CoP are as expected from datasheet
    assert pyo.value(m.fs.hp.coefficient_of_performance) == pytest.approx(1.9, rel=0.05, abs=0.2)
    assert pyo.value(m.fs.hp.sink.heat[0]) == pytest.approx(720e3, rel=0.05, abs=50e3)
    
    m.fs.hp.report()
    

def test_heat_pump_with_work_and_efficiency():
    m = _make_case()
    # Fix efficiency to the Ovolondo heat pump
    m.fs._ = pyo.Constraint(expr=m.fs.hp.efficiency == 0.3305 + 0.2723 * pyo.tanh((m.fs.hp.delta_temperature_lift[0] - 57.64) / 74.19))
    m.fs.hp.approach_temperature.fix(10)  # K
    # Fix mechanical work to the example
    m.fs.hp.work_mechanical.fix(380e3)  # W
    
    assert degrees_of_freedom(m) == 0
    
    _assert_solution(m)
    
def test_heat_pump_with_sink_outlet_and_efficiency():
    m = _make_case()
    
    # Fix efficiency to the Ovolondo heat pump
    m.fs._ = pyo.Constraint(expr=m.fs.hp.efficiency == 0.3305 + 0.2723 * pyo.tanh((m.fs.hp.delta_temperature_lift[0] - 57.64) / 74.19))
    m.fs.hp.approach_temperature.fix(10)  # K
    # Fix sink outlet temperature to the Ovolondo heat pump
    T_sink_out = (171 + 273.15) * pyo.units.K  # K
    m.fs.hp.sink_outlet.enth_mol.fix(
        m.fs.water.htpx(T_sink_out , m.fs.hp.sink_inlet.pressure[0])
    )  # J/mol
    
    assert degrees_of_freedom(m) == 0
    
    _assert_solution(m)
