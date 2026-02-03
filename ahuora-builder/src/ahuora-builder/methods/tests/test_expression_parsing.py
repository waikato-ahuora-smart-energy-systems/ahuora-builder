from pyomo.environ import ConcreteModel, Var, value, Reference
from ..expression_parsing import parse_expression
from ...properties_manager import PropertiesManager


def test_parse_expression():
    m = ConcreteModel()
    m.x = Var(initialize=2)
    m.properties_map = PropertiesManager()
    m.properties_map.add(
        123, Reference(m.x), "m.x"
    )

    expr = parse_expression("id_123 + 2", m,1)
    assert value(expr) == 4
