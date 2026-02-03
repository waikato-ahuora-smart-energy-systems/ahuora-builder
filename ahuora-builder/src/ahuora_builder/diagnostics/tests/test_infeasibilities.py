import pyomo.environ as pyo
from ..infeasibilities import compute_infeasibilities, print_infeasibilities
from ...properties_manager import PropertiesManager



def test_infeasibilities(capsys):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=5.0)
    model.c = pyo.Constraint(expr=model.x == 10.0)

    properties_map = PropertiesManager()
    properties_map.add(
        12345,
        pyo.Reference(model.x),
        "property name"
    )
    properties_map.add_constraint(
        12345,
        [model.c])


    print_infeasibilities(properties_map)
    captured = capsys.readouterr()
    assert "property name" in captured.out
    assert "5.0" in captured.out


