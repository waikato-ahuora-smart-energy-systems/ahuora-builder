import pytest
 
from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    Expression,
    Param,
    units as pyunits,
    value,
    Var,
)
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent
 
from idaes.core import FlowsheetBlock
from idaes.models.unit_models.heat_exchanger_ntu import (
    HeatExchangerNTU as HXNTU,
    HXNTUInitializer,
)
 
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models_extra.column_models.properties.MEA_solvent import (
    configuration as aqueous_mea,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.core.util.testing import initialization_tester, PhysicalParameterTestBlock
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)
from idaes.core.util import DiagnosticsToolbox    
from idaes.models.properties import iapws95

def test_ntu_hx():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = iapws95.Iapws95ParameterBlock(
    phase_presentation=iapws95.PhaseType.MIX,
    state_vars=iapws95.StateVars.PH
    )
    import math
    
    m.fs.unit = HXNTU(
        hot_side={
            "property_package": m.fs.properties,
            "has_pressure_change": False,
        },
        cold_side={
            "property_package": m.fs.properties,
            "has_pressure_change": False,
        },
    )
    
    # Hot fluid
    m.fs.unit.hot_side_inlet.flow_mol[0].fix(60.54879)
    m.fs.unit.hot_side_inlet.enth_mol[0].fix(m.fs.properties.htpx(p=202650*pyunits.Pa, T=392.23*pyunits.K))
    #m.fs.unit.hot_side_inlet.temperature[0].fix(392.23)
    m.fs.unit.hot_side_inlet.pressure[0].fix(202650)
    
    
    # Cold fluid
    m.fs.unit.cold_side_inlet.flow_mol[0].fix(63.01910)
    m.fs.unit.cold_side_inlet.enth_mol[0].fix(m.fs.properties.htpx(p=202650*pyunits.Pa, T=326.36*pyunits.K))
    #m.fs.unit.cold_side_inlet.temperature[0].fix(326.36)
    m.fs.unit.cold_side_inlet.pressure[0].fix(202650)
    
    
    # Unit design variables
    m.fs.unit.area.fix(100)
    m.fs.unit.heat_transfer_coefficient.fix(200)
    e = 0.7
    m.fs.unit.effectiveness.fix(e)
    
    # m.fs.unit.hot_side.deltaP.fix(-2000)
    # m.fs.unit.cold_side.deltaP.fix(-2000)
    
    m.fs.unit.initialize()
    
    solver = get_solver()
    results = solver.solve(m, tee=True)
    
    m.fs.unit.report()
    print(value(m.fs.unit.NTU[0]))
    print(value(m.fs.unit.heat_duty[0]))
    print(value(m.fs.unit.Cmin[0]))
    print(value(m.fs.unit.Cmax[0]))
    
    if value(m.fs.unit.hot_side.properties_in[0].flow_mol * m.fs.unit.hot_side.properties_in[0].cp_mol) < value(m.fs.unit.cold_side.properties_in[0].flow_mol * m.fs.unit.cold_side.properties_in[0].cp_mol):
        Cmin = value(
            m.fs.unit.hot_side.properties_in[0].flow_mol
            * m.fs.unit.hot_side.properties_in[0].cp_mol
        )
        Cmax = value(
            m.fs.unit.cold_side.properties_in[0].flow_mol
            * m.fs.unit.cold_side.properties_in[0].cp_mol
        )
    else:
        Cmax = value(
            m.fs.unit.hot_side.properties_in[0].flow_mol
            * m.fs.unit.hot_side.properties_in[0].cp_mol
        )
        Cmin = value(
                m.fs.unit.hot_side.properties_in[0].flow_mol
                * m.fs.unit.hot_side.properties_in[0].cp_mol
            )
    
    NTU = value(m.fs.unit.heat_transfer_coefficient[0] * m.fs.unit.area / Cmin)
    Q_duty = value(Cmin * e * (m.fs.unit.hot_side.properties_in[0].temperature - m.fs.unit.cold_side.properties_in[0].temperature))
    # Cmin = 5
    # Cmax = 5
    #NTU = 18
    #Q_duty = 20
    assert math.isclose(value(m.fs.unit.Cmin[0]), Cmin, rel_tol=1e-5), (
        f"Cmin mismatch: model={value(m.fs.unit.Cmin[0])}, calculated={Cmin}"
    )
    
    assert math.isclose(value(m.fs.unit.Cmax[0]), Cmax, rel_tol=1e-5), (
        f"Cmax mismatch: model={value(m.fs.unit.Cmax[0])}, calculated={Cmax}"
    )
    assert math.isclose(value(m.fs.unit.NTU[0]), NTU, rel_tol=1e-5), (
        f"NTU mismatch: model={value(m.fs.unit.NTU[0])}, calculated={NTU}"
    )
    assert math.isclose(value(m.fs.unit.heat_duty[0]), Q_duty, rel_tol=1e-5), (
        f"Heat duty mismatch: model={value(m.fs.unit.heat_duty[0])}, calculated={Q_duty}"
    )