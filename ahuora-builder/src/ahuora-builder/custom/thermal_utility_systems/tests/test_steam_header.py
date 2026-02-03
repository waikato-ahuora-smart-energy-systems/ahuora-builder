from ahuora_builder.custom.thermal_utility_systems.steam_header import SteamHeader
from ahuora_builder.custom.custom_heater import DynamicHeater
import pytest
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from property_packages.build_package import build_package
from pyomo.network import Arc, SequentialDecomposition
import idaes.logger as idaeslog



@pytest.fixture
def steam_header_base_case():
    def _make_case(num_inlets=2, num_outlets=3):
        # This defines the base case for all tests
        m = pyo.ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])
        # Create CustomHeader with 2 inlets and 3 outlets
        m.fs.header = SteamHeader(property_package=m.fs.water, num_inlets=num_inlets, num_outlets=num_outlets)

        # Set inlet conditions for inlet_1
        m.fs.header.mixer.inlet_1.flow_mol.fix(1000)  # mol/s
        m.fs.header.mixer.inlet_1.pressure.fix(201325)  # Pa
        m.fs.header.mixer.inlet_1.enth_mol.fix(m.fs.water.htpx((300 + 273.15) * pyo.units.K, m.fs.header.mixer.inlet_1.pressure[0]))  # J/mol

        # Initialise outlet flow for the let-down line (i.e., outlet_2 in this case)
        vent = m.fs.header.splitter.vent
        vent.flow_mol.fix(0)
        vent.flow_mol.unfix()
        return m
    return _make_case

def assert_header_solution(
        m,
        expected_total_flow,
        expected_vent_flow,
        expected_makeup_flow,
        expected_liquid_flow,
        has_heat_loss,
        has_pressure_loss,
    ):
    eps_rel = 1e-4 
    eps_abs = 1e-4
    # Check degrees of freedom
    assert degrees_of_freedom(m) == 0
    
    # Initialize the model
    m.fs.header.initialize()
    
    # Solve the model
    opt = pyo.SolverFactory("ipopt")
    results = opt.solve(m, tee=False)
    
    # Check that solve was successful
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    
    # Verify mass balance: total inlet flow = total vapor outlet flow + liquid outlet flow
    total_inlet_flow = sum([pyo.value(inlet_sb[0].flow_mol) for inlet_sb in m.fs.header.inlet_blocks]) + pyo.value(m.fs.header.makeup_flow_mol[0])
    total_outlet_flow = sum([pyo.value(outlet_sb[0].flow_mol) for outlet_sb in m.fs.header.outlet_blocks])
    condensate_flow = pyo.value(m.fs.header.phase_separator.condensate_outlet.flow_mol[0])
    assert expected_total_flow == pytest.approx(total_outlet_flow + condensate_flow, rel=eps_rel, abs=eps_abs)
    assert expected_total_flow == pytest.approx(total_inlet_flow, rel=eps_rel, abs=eps_abs)
    
    assert expected_liquid_flow == pytest.approx(condensate_flow, rel=eps_rel, abs=eps_abs)
    assert expected_vent_flow == pytest.approx(pyo.value(m.fs.header.splitter.vent.flow_mol[0]), rel=eps_rel, abs=0.02)
    assert expected_makeup_flow == pytest.approx(pyo.value(m.fs.header.makeup_flow_mol[0]), rel=eps_rel, abs=0.02)

    # Verify pressure drop is applied correctly
    if pyo.value(m.fs.header.mixer.inlet_1.pressure[0]) > eps_rel:
        expected_min_inlet_pressure = pyo.value(m.fs.header.mixer.inlet_1.pressure[0])
    else:
        expected_min_inlet_pressure = pyo.value(m.fs.header.mixer.inlet_1.pressure[0])
    expected_outlet_pressure = expected_min_inlet_pressure + pyo.value(m.fs.header.deltaP[0])  # Accounts for pressure drop
    
    assert pyo.value(m.fs.header.splitter.outlet_1.pressure[0]) == pytest.approx(expected_outlet_pressure, rel=eps_rel, abs=eps_abs)
    assert pyo.value(m.fs.header.splitter.vent.pressure[0]) == pytest.approx(expected_outlet_pressure, rel=eps_rel, abs=eps_abs)

    assert pyo.value(m.fs.header.splitter.outlet_1.pressure[0]) == pytest.approx(pyo.value(m.fs.header.phase_separator.condensate_outlet.pressure[0]), rel=eps_rel, abs=eps_abs)
    assert pyo.value(m.fs.header.splitter.vent.pressure[0]) == pytest.approx(pyo.value(m.fs.header.phase_separator.condensate_outlet.pressure[0]), rel=eps_rel, abs=eps_abs)

    cooler_inlet_pressure = pyo.value(m.fs.header.mixer.mixed_state[0].pressure)
    cooler_outlet_pressure = pyo.value(m.fs.header.pressure[0])
    cooler_deltaP = -pyo.value(m.fs.header.deltaP[0])
    if total_outlet_flow > eps_abs:
        assert cooler_outlet_pressure == pytest.approx(cooler_inlet_pressure - cooler_deltaP, rel=eps_rel, abs=eps_abs)
    
    # Verify molar enthalpy relationships
    assert pyo.value(m.fs.header.splitter.outlet_1.enth_mol[0]) == pytest.approx(pyo.value(m.fs.header.splitter.vent.enth_mol[0]), rel=eps_rel, abs=eps_abs)

    hV = pyo.value(m.fs.header.phase_separator.vapor_outlet.enth_mol[0])  # J/mol
    hL = pyo.value(m.fs.header.phase_separator.condensate_outlet.enth_mol[0])  # J/mol
    if pyo.value(m.fs.header.phase_separator.condensate_outlet.flow_mol[0]) > eps_abs:
        assert hV >= hL

    # Verify cooling effect
    cooler_inlet_temperature = pyo.value(m.fs.header.mixer.mixed_state[0].temperature)
    cooler_outlet_temperature = pyo.value(m.fs.header.temperature[0])
    if total_outlet_flow > eps_abs:
        if has_heat_loss:
            assert cooler_outlet_temperature < cooler_inlet_temperature
        else:
            assert cooler_outlet_temperature == pytest.approx(cooler_inlet_temperature, rel=eps_rel, abs=eps_abs)

    cooler_inlet_total_energy = pyo.value(m.fs.header.mixer.mixed_state[0].enth_mol * m.fs.header.mixer.mixed_state[0].flow_mol)
    cooler_outlet_total_energy = pyo.value(m.fs.header.enth_mol[0] * m.fs.header.total_flow_mol[0])
    cooler_duty = -pyo.value(m.fs.header.heat_duty[0])
    assert cooler_outlet_total_energy == pytest.approx(cooler_inlet_total_energy - cooler_duty, rel=eps_rel, abs=eps_abs)

    # Test that the unit has the expected structure
    assert hasattr(m.fs.header, 'mixer')
    assert hasattr(m.fs.header, 'cooler')
    assert hasattr(m.fs.header, 'phase_separator')
    assert hasattr(m.fs.header, 'splitter')
    assert hasattr(m.fs.header.phase_separator, 'condensate_outlet')
    
    # Test that ports exist
    assert hasattr(m.fs.header.mixer, 'inlet_1')
    assert hasattr(m.fs.header.splitter, 'outlet_1')
    assert hasattr(m.fs.header.splitter, 'vent')

    inlet_mass_list = [pyo.value(i[0].flow_mass) for i in m.fs.header.inlet_blocks] + [ pyo.value(m.fs.header.makeup_flow_mol[0] * 0.01801528)]  # convert mol/s to kg/s
    outlet_mass_list = [pyo.value(i[0].flow_mass) for i in m.fs.header.outlet_blocks]
    condensate_mass = pyo.value(m.fs.header.phase_separator.condensate_outlet_state[0].flow_mass)
    # Test all mass flows are non-zero and sum of inlet and outlet masses is approx same
    assert sum(inlet_mass_list) == pytest.approx(sum(outlet_mass_list) + condensate_mass, rel=eps_rel, abs=eps_abs)
    
    min_inlet_pressure = min([pyo.value(i[0].pressure) for i in m.fs.header.inlet_blocks])
    min_outlet_pressure = min([pyo.value(i[0].pressure) for i in m.fs.header.outlet_blocks])
    min_outlet_pressure = min(pyo.value(m.fs.header.phase_separator.condensate_outlet.pressure[0]),  min_outlet_pressure)
    
    # Test minimum input pressure is >= outlet pressure and all pressure are non-zero
    assert min_inlet_pressure >= min_outlet_pressure - eps_abs
    assert min_inlet_pressure > eps_abs
    assert min_outlet_pressure > eps_abs

def test_header_with_inlet_greater_than_outlet_flow(steam_header_base_case):
    # Case where there is zero liquid flow and there is more known inlet than outlet flows
    m = steam_header_base_case(num_inlets=1, num_outlets=1)
    
    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(800)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(-1000)  # W -> heat loss
    m.fs.header.deltaP[0].fix(-1000)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1000,
        expected_vent_flow=200,
        expected_makeup_flow=0,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_condensate_in_inlet_flow(steam_header_base_case):
    # Case where there is zero liquid flow and there is more known inlet than outlet flows
    m = steam_header_base_case(num_inlets=1, num_outlets=1)

    h_val = pyo.value(m.fs.water.htpx(p=201325 * pyo.units.Pa, x=0.9))
    m.fs.header.mixer.inlet_1.enth_mol.fix(h_val)  # J/mol

    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(800)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1000,
        expected_vent_flow=100,
        expected_makeup_flow=0,
        expected_liquid_flow=100,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_inlet_equal_to_outlet_flow(steam_header_base_case):
    # Case where there is zero liquid flow and the header is balanced
    m = steam_header_base_case(num_inlets=1, num_outlets=1)
    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(1000)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1000,
        expected_vent_flow=0,
        expected_makeup_flow=0,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_inlet_less_than_outlet_flow(steam_header_base_case):
    # Case where there is zero liquid flow and the header is balanced
    m = steam_header_base_case(num_inlets=1, num_outlets=1)
    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(1200)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1200,
        expected_vent_flow=0,
        expected_makeup_flow=200,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_zero_flow(steam_header_base_case):
    # Case where there is zero liquid flow and the header is balanced
    m = steam_header_base_case(num_inlets=1, num_outlets=1)
    m.fs.header.mixer.inlet_1.flow_mol.fix(0)  # mol/s
    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(0)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=0,
        expected_vent_flow=0,
        expected_makeup_flow=0,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_inlet_greater_than_three_defined_outlet_flows(steam_header_base_case):
    # Case where there is zero liquid flow and there is more known inlet than outlet flows
    m = steam_header_base_case(num_inlets=1, num_outlets=3)
    
    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(500)  # mol/s
    m.fs.header.splitter.outlet_2.flow_mol.fix(100)  # mol/s
    m.fs.header.splitter.outlet_3.flow_mol.fix(200)  # mol/s

    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(-1000)  # W -> heat loss
    m.fs.header.deltaP[0].fix(-1000)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1000,
        expected_vent_flow=200,
        expected_makeup_flow=0,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_inlet_less_than_two_defined_outlet_flows(steam_header_base_case):
    # Case where there is zero liquid flow and the header is balanced
    m = steam_header_base_case(num_inlets=1, num_outlets=2)
    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(600)  # mol/s
    m.fs.header.splitter.outlet_2.flow_mol.fix(600)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1200,
        expected_vent_flow=0,
        expected_makeup_flow=200,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_header_with_two_defined_inlet_less_than_two_defined_outlet_flows(steam_header_base_case):
    # Case where there is zero liquid flow and the header is balanced
    m = steam_header_base_case(num_inlets=2, num_outlets=2)
    m.fs.header.mixer.inlet_2.flow_mol.fix(100)  # mol/s
    m.fs.header.mixer.inlet_2.pressure.fix(221325)  # Pa
    m.fs.header.mixer.inlet_2.enth_mol.fix(m.fs.water.htpx((310 + 273.15) * pyo.units.K, m.fs.header.mixer.inlet_1.pressure[0]))  # J/mol

    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(600)  # mol/s
    m.fs.header.splitter.outlet_2.flow_mol.fix(600)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    assert_header_solution(
        m,
        expected_total_flow=1200,
        expected_vent_flow=0,
        expected_makeup_flow=100,
        expected_liquid_flow=0,
        has_heat_loss=True if pyo.value(m.fs.header.heat_duty[0]) < 0 else False,
        has_pressure_loss=True if pyo.value(m.fs.header.deltaP[0]) < 0 else False,
    )

def test_sequential_decomposition(steam_header_base_case):
    m = steam_header_base_case(num_inlets=1, num_outlets=1)

    m.fs.heater = DynamicHeater(property_package=m.fs.water)

    # Set user steam demand
    m.fs.header.splitter.outlet_1.flow_mol.fix(600)  # mol/s
    # Set cooler specifications
    m.fs.header.heat_duty[0].fix(0)  # W -> heat loss
    m.fs.header.deltaP[0].fix(0)  # Pa -> pressure loss

    m.fs.heater.inlet.flow_mol.fix(100)
    m.fs.heater.inlet.enth_mol.fix(m.fs.water.htpx((300 + 273.15) * pyo.units.K, m.fs.header.mixer.inlet_1.pressure[0]))  # J/mol
    m.fs.heater.inlet.pressure.fix(101325)

    m.fs.heater.heat_duty.fix(0)

    m.fs.header.mixer.inlet_1.flow_mol.unfix()
    m.fs.header.mixer.inlet_1.enth_mol.unfix()
    m.fs.header.mixer.inlet_1.pressure.unfix()

    m.fs.heater_to_header = Arc(
        source=m.fs.heater.outlet, destination=m.fs.header.mixer.inlet_1
    )

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    assert degrees_of_freedom(m) == 0

    # create Sequential Decomposition object
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 1

    # create computation graph
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    # get calculation order
    order = seq.calculation_order(G)

    for o in heuristic_tear_set:
        print(o.name)

    print("Initialization order:")
    for o in order:
        print(o[0].name)

    # define unit initialisation function
    def init_unit(unit):
        unit.initialize()

    # run sequential decomposition
    seq.run(m, init_unit)

def test_header_with_inlet_steam_generator():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])

    # Add a heater before the header
    m.fs.boiler = DynamicHeater(
        property_package=m.fs.water,
        has_pressure_change=True,
    )
    # Create CustomHeader with 2 inlets and 3 outlets
    m.fs.header = SteamHeader(
        property_package=m.fs.water, 
        num_inlets=1, 
        num_outlets=1
    )
    # Connect heater outlet to header inlet_1
    m.boiler_to_header_arc = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.header.mixer.inlet_1
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Set inlet conditions for inlet_1
    m.fs.boiler.inlet.flow_mol.fix(1000)  # mol/s
    m.fs.boiler.inlet.pressure.fix(200 * 1000)  # Pa
    m.fs.boiler.inlet.enth_mol.fix(
        m.fs.water.htpx(
            (15 + 273.15) * pyo.units.K, 
            m.fs.boiler.inlet.pressure[0]
        )
    )  # J/mol
    m.fs.boiler.heat_duty.fix(
        50.577 * 1000 * 1000
    )
    m.fs.boiler.deltaP.fix(
        0.0
    )
    m.fs.header.splitter.outlet_1.flow_mol.fix(500)
    m.fs.header.heat_duty.fix(
        0.0
    )
    m.fs.header.deltaP.fix(
        0.0
    )
    
    # Initialize the heater first
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 1

    # create computation graph
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    # get calculation order
    order = seq.calculation_order(G)

    for o in heuristic_tear_set:
        print(o.name)

    for o in order:
        print(o[0].name)

    # define unit initialisation function
    def init_unit(unit):
        unit.initialize()

    # run sequential decomposition
    seq.run(m, init_unit) 

    dof = degrees_of_freedom(m)

    opt = pyo.SolverFactory("ipopt_v2")
    results = opt.solve(m, tee=False)

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal       

    assert_header_solution(
        m=m, 
        expected_total_flow=1000,
        expected_vent_flow=500,
        expected_makeup_flow=0,
        expected_liquid_flow=0,
        has_heat_loss=0,
        has_pressure_loss=0,
    )

def test_header_with_inlet_steam_generator_with_insufficent_flow():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])

    # Add a heater before the header
    m.fs.boiler = DynamicHeater(
        property_package=m.fs.water,
        has_pressure_change=True,
    )
    # Create CustomHeader with 2 inlets and 3 outlets
    m.fs.header = SteamHeader(
        property_package=m.fs.water, 
        num_inlets=1, 
        num_outlets=1
    )
    # Connect heater outlet to header inlet_1
    m.boiler_to_header_arc = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.header.mixer.inlet_1
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Set inlet conditions for inlet_1
    m.fs.boiler.inlet.flow_mol.fix(1000)  # mol/s
    m.fs.boiler.inlet.pressure.fix(200 * 1000)  # Pa
    m.fs.boiler.inlet.enth_mol.fix(
        m.fs.water.htpx(
            (15 + 273.15) * pyo.units.K, 
            m.fs.boiler.inlet.pressure[0]
        )
    )  # J/mol
    m.fs.boiler.heat_duty.fix(
        50.577 * 1000 * 1000
    )
    m.fs.boiler.deltaP.fix(
        0.0
    )
    m.fs.header.splitter.outlet_1.flow_mol.fix(1500)
    m.fs.header.heat_duty.fix(
        0.0
    )
    m.fs.header.deltaP.fix(
        0.0
    )
    
    # Initialize the heater first
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 1

    # create computation graph
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    # get calculation order
    order = seq.calculation_order(G)

    for o in heuristic_tear_set:
        print(o.name)

    for o in order:
        print(o[0].name)

    # define unit initialisation function
    def init_unit(unit):
        unit.initialize()

    # run sequential decomposition
    seq.run(m, init_unit) 

    opt = pyo.SolverFactory("ipopt_v2")
    results = opt.solve(m, tee=False)

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal       
    
    assert_header_solution(
        m=m, 
        expected_total_flow=1500,
        expected_vent_flow=0,
        expected_makeup_flow=500,
        expected_liquid_flow=0,
        has_heat_loss=0,
        has_pressure_loss=0,
    )
 
def test_header_with_inlet_steam_generator_with_losses():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])

    # Add a heater before the header
    m.fs.boiler = DynamicHeater(
        property_package=m.fs.water,
        has_pressure_change=True,
    )
    # Create CustomHeader with 2 inlets and 3 outlets
    m.fs.header = SteamHeader(
        property_package=m.fs.water, 
        num_inlets=1, 
        num_outlets=1
    )
    # Connect heater outlet to header inlet_1
    m.boiler_to_header_arc = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.header.mixer.inlet_1
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Set inlet conditions for inlet_1
    m.fs.boiler.inlet.flow_mol.fix(1000)  # mol/s
    m.fs.boiler.inlet.pressure.fix(200 * 1000)  # Pa
    m.fs.boiler.inlet.enth_mol.fix(
        m.fs.water.htpx(
            (15 + 273.15) * pyo.units.K, 
            m.fs.boiler.inlet.pressure[0]
        )
    )  # J/mol
    m.fs.boiler.heat_duty.fix(
        50.577 * 1000 * 1000
    )
    m.fs.boiler.deltaP.fix(
        0.0
    )
    m.fs.header.splitter.outlet_1.flow_mol.fix(500)
    m.fs.header.heat_duty.fix(
        -10.0
    )
    m.fs.header.deltaP.fix(
        -5000.0
    )
    
    # Initialize the heater first
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 1

    # create computation graph
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    # get calculation order
    order = seq.calculation_order(G)

    for o in heuristic_tear_set:
        print(o.name)

    for o in order:
        print(o[0].name)

    # define unit initialisation function
    def init_unit(unit):
        unit.initialize()

    # run sequential decomposition
    seq.run(m, init_unit) 

    opt = pyo.SolverFactory("ipopt_v2")
    results = opt.solve(m, tee=False)

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal       

    assert_header_solution(
        m=m, 
        expected_total_flow=1000,
        expected_vent_flow=500,
        expected_makeup_flow=0,
        expected_liquid_flow=0,
        has_heat_loss=True,
        has_pressure_loss=True,
    )
 
def test_header_with_inlet_steam_generator_with_wet_steam():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])

    # Add a heater before the header
    m.fs.boiler = DynamicHeater(
        property_package=m.fs.water,
        has_pressure_change=True,
    )
    # Create CustomHeader with 2 inlets and 3 outlets
    m.fs.header = SteamHeader(
        property_package=m.fs.water, 
        num_inlets=1, 
        num_outlets=1
    )
    # Connect heater outlet to header inlet_1
    m.boiler_to_header_arc = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.header.mixer.inlet_1
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Set inlet conditions for inlet_1
    m.fs.boiler.inlet.flow_mol.fix(1000)  # mol/s
    m.fs.boiler.inlet.pressure.fix(200 * 1000)  # Pa
    m.fs.boiler.inlet.enth_mol.fix(
        m.fs.water.htpx(
            (15 + 273.15) * pyo.units.K, 
            m.fs.boiler.inlet.pressure[0]
        )
    )  # J/mol
    m.fs.boiler.heat_duty.fix(
        27784964.63
    )
    m.fs.boiler.deltaP.fix(
        0.0
    )
    m.fs.header.splitter.outlet_1.flow_mol.fix(500)
    m.fs.header.heat_duty.fix(
        0.0
    )
    m.fs.header.deltaP.fix(
        0.0
    )
    
    # Initialize the heater first
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = 1

    # create computation graph
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")
    # get calculation order
    order = seq.calculation_order(G)

    for o in heuristic_tear_set:
        print(o.name)

    for o in order:
        print(o[0].name)

    # define unit initialisation function
    def init_unit(unit):
        unit.initialize()

    # run sequential decomposition
    seq.run(m, init_unit) 

    opt = pyo.SolverFactory("ipopt_v2")
    results = opt.solve(m, tee=False)

    assert results.solver.termination_condition == pyo.TerminationCondition.optimal       

    assert_header_solution(
        m=m, 
        expected_total_flow=1000,
        expected_vent_flow=0,
        expected_makeup_flow=0.0,
        expected_liquid_flow=500,
        has_heat_loss=0,
        has_pressure_loss=0,
    )


