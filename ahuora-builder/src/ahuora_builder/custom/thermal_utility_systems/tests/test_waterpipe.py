import pytest

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models_extra.power_generation.unit_models.waterpipe import WaterPipe
from property_packages.build_package import build_package
from idaes.core.util import DiagnosticsToolbox 

def test_waterpipe():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.water = build_package("helmholtz", ["water"], ["Liq"])
    m.fs.pipe = WaterPipe(property_package=m.fs.water)

    # Fix inlet conditions
    m.fs.pipe.inlet.flow_mol.fix(10)  
    m.fs.pipe.inlet.pressure.fix(101325)  
    # m.fs.pipe.inlet.temperature.fix(300)

    # Fix pipe parameters
    m.fs.pipe.length.fix(100)  # m
    m.fs.pipe.diameter.fix(0.5) 
    m.fs.pipe.number_of_pipes.fix(1)
    m.fs.pipe.elevation_change.fix(5) 
    m.fs.pipe.fcorrection_dp.fix(0.7)

    assert degrees_of_freedom(m.fs) == 0

    m.fs.pipe.initialize()
    solver = pyo.SolverFactory("ipopt")
    solver.options['max_iter']= 10000
    results = solver.solve(m, tee=False)

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal