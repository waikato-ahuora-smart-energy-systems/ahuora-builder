from pyomo.environ import ConcreteModel, Var, value, Reference, Expression, Constraint
from ..property_map_manipulation import update_property, free_property
from ..adapter import fix_slice
from ...properties_manager import PropertiesManager
from ahuora_builder_types.id_types import PropertyValueId

from idaes.core.util.model_statistics import degrees_of_freedom

def test_free_property():
    m = ConcreteModel()
    m.x = Var(initialize=2)
    # The extra variable and constraint here is just to make it a proper model.
    m.y = Var(initialize=3)
    m.constraint = Constraint(expr=m.x + m.y == 10)
    m.expr = Expression(expr=m.x + 1)
    properties_map = PropertiesManager()
    properties_map.add(
        PropertyValueId(123), Reference(m.x), "m.x"
    )
    properties_map.add(
        PropertyValueId(101), Reference(m.expr), "m.expr"
    )

    assert degrees_of_freedom(m) == 1
    property_component = properties_map.get(PropertyValueId(123))
    constraints = fix_slice(property_component.component, [5])
    properties_map.add_constraint(PropertyValueId(123), constraints)
    assert degrees_of_freedom(m) == 0

    # Test with fixing and unfixing a variable
    assert m.x.is_fixed()
    free_property(property_component)
    assert degrees_of_freedom(m) == 1
    assert not m.x.is_fixed()

    # Test with fixing and unfixing an expression
    expr_component = properties_map.get(PropertyValueId(101))
    constraints = fix_slice(expr_component.component, [10])
    properties_map.add_constraint(PropertyValueId(101), constraints)
    assert degrees_of_freedom(m) == 0
    free_property(expr_component)
    assert degrees_of_freedom(m) == 1




    