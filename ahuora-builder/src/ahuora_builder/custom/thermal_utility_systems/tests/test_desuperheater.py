import pytest

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from ahuora_builder.custom.thermal_utility_systems.desuperheater import Desuperheater
from property_packages.build_package import build_package

eps_rel = 1e-4 
eps_abs = 1e-2

def desuperheater_base_case(deltaT_superheat, T_water, T_steam, P_steam, heat_loss, pressure_loss):
    # This defines the base case for all tests
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.water = build_package(
        "helmholtz", 
        ["water"], 
        ["Liq", "Vap"]
    )

    # Create SteamUser
    m.fs.desuperheater = Desuperheater(
        property_package=m.fs.water
    )

    # Set inlet conditions for steam inlet
    m.fs.desuperheater.inlet_steam.flow_mol.fix(1000)    # mol/s
    m.fs.desuperheater.inlet_steam.pressure.fix(P_steam)  # Pa
    m.fs.desuperheater.inlet_steam.enth_mol.fix(
        m.fs.water.htpx((T_steam + 273.15) * pyo.units.K, 
        m.fs.desuperheater.inlet_steam.pressure[0]),
    )  # J/mol

    # Define the state variables
    m.fs.desuperheater.deltaT_superheat.fix(
        deltaT_superheat * pyo.units.K
    ) # K
    m.fs.desuperheater.bfw_temperature[:].fix(
        (T_water + 273.15) * pyo.units.K
    ) # K
    m.fs.desuperheater.heat_loss[:].fix(heat_loss)  # W -> heat loss
    m.fs.desuperheater.pressure_loss[:].fix(pressure_loss)  # Pa -> pressure loss

    # Check degrees of freedom
    assert degrees_of_freedom(m.fs) == 0
    
    # Initialize the model
    m.fs.desuperheater.initialize()
    
    # Solve the model
    opt = pyo.SolverFactory("ipopt_v2")
    results = opt.solve(m, tee=False)

    return m, results
    
def test_desuperheater_with_zero_superheat_in_outlet_flow():
    P = 400*1000
    T = 200
    T_w = 80
    dT_sup = 0
    Q_loss = 0
    dP = 0
    
    m, results = desuperheater_base_case(
        deltaT_superheat=dT_sup, 
        T_water=T_w, 
        T_steam=T, 
        heat_loss=Q_loss, 
        pressure_loss=dP, 
        P_steam=P,
    )

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert (pyo.value(m.fs.desuperheater.outlet_state[0].temperature) - pyo.value(m.fs.desuperheater._int_sat_vap_state[0].temperature)) == pytest.approx(dT_sup, rel=eps_rel, abs=eps_abs)
    assert pyo.value(m.fs.desuperheater.outlet_state[0].pressure) == pytest.approx(P - dP, rel=eps_rel, abs=eps_abs)

def test_desuperheater_with_zero_superheat_in_outlet_flow_and_pressure_loss():
    # Case where there is zero liquid flow and there is more known inlet than outlet flows
    P = 400*1000
    T = 200
    T_w = 80
    dT_sup = 0
    Q_loss = 0
    dP = 1000
    
    m, results = desuperheater_base_case(
        deltaT_superheat=dT_sup, 
        T_water=T_w, 
        T_steam=T, 
        heat_loss=Q_loss, 
        pressure_loss=dP, 
        P_steam=P,
    )
    
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert (pyo.value(m.fs.desuperheater.outlet_state[0].temperature) - pyo.value(m.fs.desuperheater._int_sat_vap_state[0].temperature)) == pytest.approx(dT_sup, rel=eps_rel, abs=eps_abs)
    assert pyo.value(m.fs.desuperheater.outlet_state[0].pressure) == pytest.approx(P - dP, rel=eps_rel, abs=eps_abs)

def test_desuperheater_with_10_superheat_in_outlet_flow():
    # Case where there is zero liquid flow and there is more known inlet than outlet flows
    P = 400*1000
    T = 200
    T_w = 80
    dT_sup = 10
    Q_loss = 0
    dP = 0
    
    m, results = desuperheater_base_case(
        deltaT_superheat=dT_sup, 
        T_water=T_w, 
        T_steam=T, 
        heat_loss=Q_loss, 
        pressure_loss=dP, 
        P_steam=P,
    )

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert (pyo.value(m.fs.desuperheater.outlet_state[0].temperature) - pyo.value(m.fs.desuperheater._int_sat_vap_state[0].temperature)) == pytest.approx(dT_sup, rel=eps_rel, abs=eps_abs)
    assert pyo.value(m.fs.desuperheater.outlet_state[0].pressure) == pytest.approx(P - dP, rel=eps_rel, abs=eps_abs)

def test_desuperheater_with_5_superheat_in_outlet_flow_plus_losses():
    # Case where there is zero liquid flow and there is more known inlet than outlet flows
    P = 400*1000
    T = 200
    T_w = 80
    dT_sup = 5
    Q_loss = 10000
    dP = 5000
    
    m, results = desuperheater_base_case(
        deltaT_superheat=dT_sup, 
        T_water=T_w, 
        T_steam=T, 
        heat_loss=Q_loss, 
        pressure_loss=dP, 
        P_steam=P,
    )
    
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert (pyo.value(m.fs.desuperheater.outlet_state[0].temperature) - pyo.value(m.fs.desuperheater._int_sat_vap_state[0].temperature)) == pytest.approx(dT_sup, rel=eps_rel, abs=eps_abs)
    assert pyo.value(m.fs.desuperheater.outlet_state[0].pressure) == pytest.approx(P - dP, rel=eps_rel, abs=eps_abs)
