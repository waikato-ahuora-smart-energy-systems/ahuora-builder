import pytest

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, SolverFactory, SolverStatus, TerminationCondition, Block, TransformationFactory, units, Objective, value, Constraint, Var
from pyomo.network import SequentialDecomposition, Port, Arc


# Import IDAES libraries
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom
import idaes.logger as idaeslog
from idaes.core.util.tables import _get_state_from_port  

# Import required models
from idaes.models.unit_models import (
    Feed,
    Mixer,
    Heater,
    Compressor,
    Product,
    MomentumMixingType,
)
from idaes.models.unit_models import Separator as Splitter
from idaes.models.unit_models import Compressor, PressureChanger
from idaes.models.properties.general_helmholtz import (
    HelmholtzParameterBlock,
    PhaseType,
    StateVars,
    AmountBasis,
    )
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption, Turbine
from ahuora_builder.custom.thermal_utility_systems.willans_turbine import TurbineBase
from idaes.core.util import DiagnosticsToolbox

from pyomo.util.check_units import assert_units_consistent

@pytest.fixture
def build_model_with_inputs():
    def _make_case(calculation_method):
        # Define model components and blocks
        m = ConcreteModel()
        m.fs1 = FlowsheetBlock(dynamic=False)
        m.fs1.water = HelmholtzParameterBlock(
                        pure_component="h2o",
                        phase_presentation=PhaseType.LG,
                        state_vars=StateVars.PH,
                        amount_basis=AmountBasis.MOLE,
                        )
        
        m.fs1.turbine = TurbineBase(
            property_package=m.fs1.water, 
            calculation_method=calculation_method,
            )

        # Inputs
        n_in = 187.0 * 3.6 * 15.419020010014712 # t/h to mol/s
        P_in = 41.3 # bar g
        if m.fs1.turbine.config.calculation_method == "CT_willans": # CT model needs lower pressure
            P_out = -0.4 # bar g
        else:
            P_out = 10.4 # bar g
        T_in = 381 # C

        m.fs1.turbine.inlet.flow_mol.fix(n_in/3.6)
        m.fs1.turbine.inlet.enth_mol[0].fix(
            value(
                m.fs1.water.htpx(
                    T=(T_in+273)*units.K, 
                    p=(P_in+1)*units.bar,
                )
            )
        )
        m.fs1.turbine.inlet.pressure[0].fix((P_in+1)*units.bar) # why the [0]
        m.fs1.turbine.outlet.pressure[0].fix((P_out+1)*units.bar)

        return m
    return _make_case

def assert_willans_turbine_solution(m):
    for i in range(10):
        n_in = float(i+1) / 10 * m.fs1.turbine.willans_max_mol[0]
        m.fs1.turbine.inlet.flow_mol.fix(
            n_in
        )
        solver = SolverFactory("ipopt")
        solver.options = {"tol": 1e-3, "max_iter": 5000}
        assert degrees_of_freedom(m.fs1) == 0
        result = solver.solve(m, tee=False)
        assert result.solver.termination_condition == pyo.TerminationCondition.optimal
        assert_units_consistent(m.fs1)

def test_isentropic_turbine(build_model_with_inputs):
    m = build_model_with_inputs("isentropic")

    m.fs1.turbine.efficiency_motor.fix(1.0) 
    m.fs1.turbine.efficiency_isentropic.fix(1)

    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-3, "max_iter": 5000}
    assert degrees_of_freedom(m.fs1) == 0
    result = solver.solve(m, tee=False)
    
    assert_units_consistent(m.fs1)

    m.fs1.turbine.report()
    
    assert result.solver.termination_condition == TerminationCondition.optimal
    
    assert m.fs1.turbine.ratioP[0].value == pytest.approx(0.2695035460992908, rel=0.001)
    assert pyo.value(m.fs1.turbine.efficiency_isentropic[0]) == pytest.approx(1, rel=0.001)
    assert m.fs1.turbine.efficiency_motor[0].value == pytest.approx(1.0, rel=0.001)
    assert m.fs1.turbine.work_electrical[0].value == pytest.approx(-16638744.441924531, rel=0.001)

def test_simple_willans_turbine(build_model_with_inputs):
    m = build_model_with_inputs("simple_willans")

    m.fs1.turbine.efficiency_motor.fix(1.0) 
    m.fs1.turbine.willans_slope.fix(190*18*units.J/units.mol)  # Willans slope 
    m.fs1.turbine.willans_intercept.fix(0.1366*1000*units.W)  # Willans intercept
    m.fs1.turbine.willans_max_mol.fix(100*15.4*units.mol / units.s)  # Willans intercept

    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-3, "max_iter": 5000}
    assert degrees_of_freedom(m.fs1) == 0
    result = solver.solve(m, tee=False)
    
    assert_units_consistent(m.fs1)

    m.fs1.turbine.report()
    
    assert result.solver.termination_condition == TerminationCondition.optimal

    assert m.fs1.turbine.ratioP[0].value == pytest.approx(0.2695035460992908, rel=0.001)
    assert pyo.value(m.fs1.turbine.efficiency_isentropic[0]) == pytest.approx(0.5926537201367845, rel=0.001)
    assert m.fs1.turbine.efficiency_motor[0].value == pytest.approx(1.0, rel=0.001)
    assert m.fs1.turbine.work_electrical[0].value == pytest.approx(-9861013.778938433, rel=0.001)
    assert m.fs1.turbine.willans_slope[0].value == pytest.approx(3420.0, rel=0.001)
    assert m.fs1.turbine.willans_intercept[0].value == pytest.approx(136.6, rel=0.001)
    assert m.fs1.turbine.willans_max_mol[0].value == pytest.approx(1540.0, rel=0.001)

    assert_willans_turbine_solution(m)

def test_part_load_willans_turbine(build_model_with_inputs):
    
    m = build_model_with_inputs("part_load_willans")

    m.fs1.turbine.efficiency_motor.fix(1.0) 
    m.fs1.turbine.willans_max_mol.fix(217.4*15.4)  #
    m.fs1.turbine.willans_a.fix(1.5435)  # Willans slope 
    m.fs1.turbine.willans_b.fix(0.2*units.kW)  # Willans intercept
    m.fs1.turbine.willans_efficiency.fix(1 / (0.3759 + 1))  # Willans intercept

    m.fs1.turbine.report()

    assert_willans_turbine_solution(m)

def test_tsat_willans_turbine(build_model_with_inputs):
    m = build_model_with_inputs("Tsat_willans")

    m.fs1.turbine.efficiency_motor.fix(1.0) 
    m.fs1.turbine.willans_max_mol.fix(217.4*15.4)     

    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-3, "max_iter": 5000}
    assert degrees_of_freedom(m.fs1) == 0
    result = solver.solve(m, tee=False)

    from pyomo.util.check_units import assert_units_consistent
    import pytest
    
    assert_units_consistent(m.fs1)

    m.fs1.turbine.report()
    
    assert result.solver.termination_condition == TerminationCondition.optimal
    
    assert m.fs1.turbine.ratioP[0].value == pytest.approx(0.2695035460992908, rel=0.001)
    assert pyo.value(m.fs1.turbine.efficiency_isentropic[0]) == pytest.approx(0.8000887976702635, rel=0.001)
    assert m.fs1.turbine.efficiency_motor[0].value == pytest.approx(1.0, rel=0.001)
    assert m.fs1.turbine.work_electrical[0].value == pytest.approx(-13312473.036723418, rel=0.001)
    assert m.fs1.turbine.willans_slope[0].value == pytest.approx(5724.721486921732, rel=0.001)
    assert m.fs1.turbine.willans_intercept[0].value == pytest.approx(3194420.3120209095, rel=0.001)
    assert m.fs1.turbine.willans_max_mol[0].value == pytest.approx(3347.96, rel=0.001)
    assert m.fs1.turbine.willans_a[0].value == pytest.approx(1.1916052938465973, rel=0.001)
    assert m.fs1.turbine.willans_b[0].value == pytest.approx(287807.42187938065, rel=0.001)
    assert m.fs1.turbine.willans_efficiency[0].value == pytest.approx(0.83333, rel=0.001)

    assert_willans_turbine_solution(m)

def test_bpst_willians_turbine(build_model_with_inputs):
    m = build_model_with_inputs("BPST_willans")

    m.fs1.turbine.efficiency_motor.fix(1.0) 
    m.fs1.turbine.willans_max_mol.fix(217.4*15.4) 

    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-3, "max_iter": 5000}
    assert degrees_of_freedom(m.fs1) == 0
    result = solver.solve(m, tee=False)

    from pyomo.util.check_units import assert_units_consistent
    import pytest
    
    assert_units_consistent(m.fs1)

    m.fs1.turbine.report()
    
    assert result.solver.termination_condition == TerminationCondition.optimal
    
    assert m.fs1.turbine.ratioP[0].value == pytest.approx(0.2695035460992908, rel=0.001)
    assert pyo.value(m.fs1.turbine.efficiency_isentropic[0]) == pytest.approx(0.7640203470049349, rel=0.001)
    assert m.fs1.turbine.efficiency_motor[0].value == pytest.approx(1.0, rel=0.001)
    assert m.fs1.turbine.work_electrical[0].value == pytest.approx(-12712339.29641526, rel=0.001)
    assert m.fs1.turbine.willans_slope[0].value == pytest.approx(5511.348537597738, rel=0.001)
    assert m.fs1.turbine.willans_intercept[0].value == pytest.approx(3179303.3709413796, rel=0.001)
    assert m.fs1.turbine.willans_max_mol[0].value == pytest.approx(3347.96, rel=0.001)
    assert m.fs1.turbine.willans_a[0].value == pytest.approx(1.2284271712000001, rel=0.001)
    assert m.fs1.turbine.willans_b[0].value == pytest.approx(558672.9707596999, rel=0.001)
    assert m.fs1.turbine.willans_efficiency[0].value == pytest.approx(0.8276966055721292, rel=0.001)

    assert_willans_turbine_solution(m)

def test_ct_willians_turbine(build_model_with_inputs):
    m = build_model_with_inputs("CT_willans")

    m.fs1.turbine.inlet.enth_mol[0].fix(
        value(
            m.fs1.water.htpx(
                T=(430 + 273.15) * units.K, 
                p=6000 * units.kPa,
            )
        )
    )
    m.fs1.turbine.inlet.flow_mol[0].fix(
        2000 * units.mol / units.s
    )
    m.fs1.turbine.inlet.pressure[0].fix(
        6000 * units.kPa
    )
    m.fs1.turbine.outlet.pressure[0].fix(
        6 * units.kPa
    )

    m.fs1.turbine.efficiency_motor.fix(1.0) 
    m.fs1.turbine.willans_max_mol.fix(2000) 

    solver = SolverFactory("ipopt")
    solver.options = {"tol": 1e-3, "max_iter": 5000}
    assert degrees_of_freedom(m.fs1) == 0
    result = solver.solve(m, tee=False)
    
    assert_units_consistent(m.fs1)