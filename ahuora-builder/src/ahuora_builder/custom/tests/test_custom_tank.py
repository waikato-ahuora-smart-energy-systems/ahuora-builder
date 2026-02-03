from ..custom_tank import DynamicTank
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

def test_dynamic_tank():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=True,time_set=[0,1,2,3],time_units=pyo.units.s)
    m.fs.water = build_package("helmholtz", ["water"],["Liq","Vap"])
    m.fs.tank = DynamicTank(property_package=m.fs.water,    tank_type="vertical_cylindrical_tank", has_holdup=True,
    has_heat_transfer=True, dynamic=True)

    m.discretizer = pyo.TransformationFactory("dae.finite_difference")
    m.discretizer.apply_to(m, nfe=3, wrt=m.fs.time, scheme="BACKWARD")

    # Initialisation - doen't really matter
    m.fs.tank.control_volume.material_holdup[:,:,:].fix(0.5)
    m.fs.tank.control_volume.material_holdup[:,:,:].unfix()

    # Setting the required properties

    #Inlet state - all timesteps
    m.fs.tank.inlet.flow_mol.fix(100)
    m.fs.tank.inlet.enth_mol.fix(m.fs.water.htpx(300 * pyo.units.K,101325* pyo.units.Pa)) # around 5 deg c
    m.fs.tank.inlet.pressure.fix(101325)

    #Tank properties
    m.fs.tank.tank_diameter.fix(0.5) # m
    m.fs.tank.heat_duty.fix(0)  # W
    m.fs.tank.outlet.flow_mol.fix(10)
    
    #Initial conditions
    m.fs.tank.initial_temperature.fix(300)
    m.fs.tank.initial_pressure.fix(101325)
    m.fs.tank.initial_holdup.fix(1000)

    assert degrees_of_freedom(m.fs) == 0

    m.fs.tank.initialize()


    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(m, tee=False)
    #m.fs.tank.display()
    
    b = m.fs.tank

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    
    # print("Enthalpy Liquid:   Enthalpy Vap:   Material Holdup Liquid: Material Holdup Vap")
    # for t in m.fs.time:
    #     print(m.fs.tank.enthalpy[t,"Liq"].value, m.fs.tank.enthalpy[t,"Vap"].value, pyo.value(m.fs.tank.material_phase_holdup[t,"Liq"]), pyo.value(m.fs.tank.material_phase_holdup[t,"Vap"]))

    # for t in m.fs.time:
    #     print(m.fs.tank.outlet.enth_mol[t].value, m.fs.tank.outlet.flow_mol[t].value, pyo.value(m.fs.tank.control_volume.properties_out[t].temperature), pyo.value(m.fs.tank.control_volume.properties_in[t].temperature))
        

    assert pyo.value(b.outlet.enth_mol[0]) == pytest.approx(pyo.value(b.inlet.enth_mol[0]), rel=1e-1)



if __name__ == "__main__":
    test_dynamic_tank()