import pytest

from ahuora_builder.custom.thermal_utility_systems.steam_user import SteamUser
import pyomo.environ as pyo
from pyomo.environ import (
    units as UNIT,
    value,
)
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from property_packages.build_package import build_package

@pytest.fixture
def steam_user_base_case():
    def _make_case(
            steam_flow,
            steam_pressure,
            steam_temperature,
            cond_return_rate,
            cond_return_temperature,
            deltaT_subcool,
            heat_loss,
            pressure_loss,
            has_desuperheating=False, 
            bfw_temperature=0,
            deltaT_superheat=0,
        ):
        # This defines the base case for all tests
        m = pyo.ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])
       
       # Create Steam User
        m.fs.user = SteamUser(
            property_package=m.fs.water,
            has_desuperheating=has_desuperheating,
        )
        m.fs.user.inlet_steam.flow_mol.fix(
            steam_flow
        )
        m.fs.user.inlet_steam.pressure.fix(
            steam_pressure
        ) 
        m.fs.user.inlet_steam.enth_mol.fix(
            m.fs.water.htpx(
                T=steam_temperature, 
                p=m.fs.user.inlet_steam_state[0].pressure,
            )
        ) 
        m.fs.user.cond_return_rate.fix(
            cond_return_rate
        )
        m.fs.user.cond_return_temperature.fix(
            cond_return_temperature
        )        
        m.fs.user.deltaT_subcool.fix(
            deltaT_subcool
        )
        m.fs.user.heat_loss.fix(
            heat_loss
        )
        m.fs.user.pressure_loss.fix(
            pressure_loss
        )
        if has_desuperheating:
            m.fs.user.bfw_temperature.fix(
                bfw_temperature
            )
            m.fs.user.deltaT_superheat.fix(
                deltaT_superheat
            )

        dof = degrees_of_freedom(m.fs)
        if not(dof == 0):
            raise ValueError("DoF is not zero.")

        return m
    return _make_case

def assert_user_solution(
        m,
        steam_flow,
        steam_pressure,
        steam_temperature,
        cond_return_rate,
        cond_return_temperature,
        deltaT_subcool,
        heat_loss,
        pressure_loss,
        has_desuperheating=False, 
        bfw_temperature=0,
        deltaT_superheat=0,
        ):
    eps_rel = 1e-4 
    eps_abs = 1e-4

    # Check degrees of freedom
    assert degrees_of_freedom(m.fs) == 0
    
    # Initialize the model
    m.fs.user.initialize()
    
    # Solve the model
    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(m, tee=False)
    
    # Check that solve was successful
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    
    # Verify mass balance: total inlet flow = total vapor outlet flow + liquid outlet flow
    if value(steam_flow):
        assert 0 == pytest.approx(value(m.fs.user.inlet_steam_state[0].flow_mol - steam_flow), rel=eps_rel, abs=eps_abs)
    assert 0 == pytest.approx(value(m.fs.user.inlet_steam_state[0].pressure - steam_pressure), rel=eps_rel, abs=eps_abs)
    assert 0 == pytest.approx(value(m.fs.user.inlet_steam_state[0].temperature - steam_temperature), rel=eps_rel, abs=0.1)
    
    total_flow_in = sum([value(o[0].flow_mol) for o in m.fs.user.inlet_blocks])
    total_flow_out = sum([value(o[0].flow_mol) for o in m.fs.user.outlet_blocks])
    assert total_flow_in == pytest.approx(total_flow_out, rel=eps_rel, abs=eps_abs)

    assert value(cond_return_rate * total_flow_in) == pytest.approx(value(m.fs.user.outlet_return.flow_mol[0]), rel=eps_rel, abs=eps_abs)
    assert value(cond_return_temperature) == pytest.approx(value(m.fs.user.outlet_return_state[0].temperature), rel=eps_rel, abs=eps_abs)
    assert value(deltaT_subcool) == pytest.approx(value(m.fs.user._int_outlet_sat_liq_state[0].temperature - m.fs.user._int_outlet_cond_state[0].temperature), rel=eps_rel, abs=eps_abs)
    
    assert value(heat_loss) == pytest.approx(value(
            UNIT.convert(
                m.fs.user._int_mixed_inlet_state[0].flow_mol * m.fs.user._int_mixed_inlet_state[0].enth_mol - m.fs.user.heat_demand[0] - m.fs.user._int_outlet_cond_state[0].flow_mol * m.fs.user._int_outlet_cond_state[0].enth_mol, 
                to_units=UNIT.W
            )
        ), rel=eps_rel, abs=eps_abs
    )
    assert value(pressure_loss) == pytest.approx(value(
            UNIT.convert(
                m.fs.user._int_mixed_inlet_state[0].pressure - m.fs.user._int_outlet_cond_state[0].pressure,
                to_units=UNIT.Pa
            )
        ), 
        rel=eps_rel, abs=eps_abs
    )

    if has_desuperheating:
        assert value(bfw_temperature) == pytest.approx(value(m.fs.user.inlet_water_state[0].temperature), rel=eps_rel, abs=eps_abs)
        assert value(deltaT_superheat) == pytest.approx(value(m.fs.user._int_mixed_inlet_state[0].temperature - m.fs.user._int_inlet_sat_vap_state[0].temperature), rel=eps_rel, abs=eps_abs)

def test_steam_user_flow_case(steam_user_base_case):
    args = {
        "steam_flow": 1000 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.7,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 10 * UNIT.K,
        "heat_loss": 1000 * UNIT.W,
        "pressure_loss": 5000 * UNIT.Pa,
        "has_desuperheating": False,
        "bfw_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_superheat": 10 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    assert_user_solution(m, **args)

def test_steam_user_flow_case_with_parital_desuperheating(steam_user_base_case):
    args = {
        "steam_flow": 1000 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.7,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 10 * UNIT.K,
        "heat_loss": 1000 * UNIT.W,
        "pressure_loss": 5000 * UNIT.Pa,
        "has_desuperheating": True,
        "bfw_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_superheat": 10 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    assert_user_solution(m, **args)

def test_steam_user_flow_case_with_full_desuperheating(steam_user_base_case):
    args = {
        "steam_flow": 1000 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.7,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 10 * UNIT.K,
        "heat_loss": 1000 * UNIT.W,
        "pressure_loss": 5000 * UNIT.Pa,
        "has_desuperheating": True,
        "bfw_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_superheat": 0 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    assert_user_solution(m, **args)

def test_steam_user_demand_case(steam_user_base_case):
    args = {
        "steam_flow": 1000 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.7,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 0 * UNIT.K,
        "heat_loss": 0 * UNIT.W,
        "pressure_loss": 0 * UNIT.Pa,
        "has_desuperheating": False,
        "bfw_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_superheat": 10 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    m.fs.user.heat_demand.fix(
        5 * 1000 * 1000 * UNIT.W
    )
    m.fs.user.inlet_steam.flow_mol[0].unfix()
    args["steam_flow"] = False
    assert_user_solution(m, **args)

def test_steam_user_demand_case_with_subcooling(steam_user_base_case):
    args = {
        "steam_flow": 0 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.7,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 10 * UNIT.K,
        "heat_loss": 0 * UNIT.W,
        "pressure_loss": 0 * UNIT.Pa,
        "has_desuperheating": False,
        "bfw_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_superheat": 10 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    m.fs.user.heat_demand.fix(
        5 * 1000 * 1000 * UNIT.W
    )
    m.fs.user.inlet_steam.flow_mol[0].unfix()
    args["steam_flow"] = False
    assert_user_solution(m, **args)

def test_steam_user_demand_case_with_subcooling_and_desuperheating(steam_user_base_case):
    args = {
        "steam_flow": 0 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.5,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 10 * UNIT.K,
        "heat_loss": 1000 * UNIT.W,
        "pressure_loss": 1000 * UNIT.Pa,
        "has_desuperheating": True,
        "bfw_temperature": (110 + 273.15) * UNIT.K,
        "deltaT_superheat": 10 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    m.fs.user.heat_demand.fix(
        5 * 1000 * 1000 * UNIT.W
    )
    m.fs.user.inlet_steam.flow_mol[0].unfix()
    args["steam_flow"] = False
    assert_user_solution(m, **args)

def test_steam_user_demand_case_with_zero_flow(steam_user_base_case):
    args = {
        "steam_flow": 0 * UNIT.mol / UNIT.s,
        "steam_pressure": 2000 * 1000 * UNIT.Pa,
        "steam_temperature": (250 + 273.15) * UNIT.K,
        "cond_return_rate": 0.5,
        "cond_return_temperature": (80 + 273.15) * UNIT.K,
        "deltaT_subcool": 0 * UNIT.K,
        "heat_loss": 0 * UNIT.W,
        "pressure_loss": 0 * UNIT.Pa,
        "has_desuperheating": False,
        "bfw_temperature": (110 + 273.15) * UNIT.K,
        "deltaT_superheat": 10 * UNIT.K,        
    }
    m = steam_user_base_case(**args)
    args["steam_flow"] = False
    assert_user_solution(m, **args)
