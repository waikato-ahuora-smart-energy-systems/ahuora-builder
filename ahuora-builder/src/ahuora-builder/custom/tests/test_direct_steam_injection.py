from ..direct_steam_injection import Dsi
import pytest
import pyomo.environ as pyo
from pyomo.network import Arc
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.models.unit_models import Heater, Valve
from idaes.models.properties import iapws95
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
# need to pip install ahuora_compounds@git+https://github.com/waikato-ahuora-smart-energy-systems/PropertyPackages.git@v0.0.25
from property_packages.build_package import build_package

def test_dsi():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.steam_properties = build_package("helmholtz", ["water"],["Liq","Vap"])
    m.fs.milk_properties = build_package("milk", ["water","milk_solid"],["Liq","Vap"])
    m.fs.dsi = Dsi(property_package=m.fs.milk_properties,steam_property_package=m.fs.steam_properties)
    
    m.fs.dsi.inlet.flow_mol.fix(1)
    m.fs.dsi.properties_milk_in[0].constrain_component(m.fs.dsi.properties_milk_in[0].temperature, 300 * pyo.units.K)
    m.fs.dsi.inlet.pressure.fix(101325)
    m.fs.dsi.inlet.mole_frac_comp[0,"water"].fix(0.9)
    m.fs.dsi.inlet.mole_frac_comp[0,"milk_solid"].fix(0.1)

    m.fs.dsi.steam_inlet.flow_mol.fix(1)
    m.fs.dsi.properties_steam_in[0].constrain_component(m.fs.dsi.properties_steam_in[0].temperature, 400 * pyo.units.K)
    m.fs.dsi.steam_inlet.pressure.fix(101325)

    m.fs.dsi.initialize()
    assert degrees_of_freedom(m.fs.dsi.properties_milk_in) == 0
    assert degrees_of_freedom(m.fs.dsi.properties_steam_in) == 0
    assert degrees_of_freedom(m.fs) == 0

    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(m, tee=True)
    
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert degrees_of_freedom(m.fs) == 0


