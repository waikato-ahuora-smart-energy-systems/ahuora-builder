from pyomo.environ import Var, Constraint, units as pyunits, Block, Reference

def add_inverted(block: Block,component_name: str):
    """
    Adds inverted variables and constraints to a block with variable component_name.
    e.g if component_name is 'deltaP', adds deltaP_inverted variable, so we can do 
    pressure drop instead of pressure increase.
    """
    inverted_component_name = f"{component_name}_inverted"

    component = getattr(block, component_name)

    inverted_component = Var(
        component.index_set(),
        # Explanation for units:
        # Reference() turns a var/expression into a indexed object, even if it's scalar.
        # next() gets the first item from the indexed object
        # This is a way of getting around the fact that an indexed component
        # will only have its units defined at the individual index level.
        units=next(Reference(component).values()).get_units(),
    )

    block.add_component(inverted_component_name, inverted_component)

    def _inverted_rule(b, *indexes):
        return inverted_component[indexes] == -component[indexes]

    constraint = Constraint(
        component.index_set(),
        rule=_inverted_rule,
        doc=f"Inverted {component_name} Constraint",
    )
    block.add_component(f"{inverted_component_name}_constraint", constraint)

def initialise_inverted(block: Block, component_name: str):
    """
    Initialises the inverted deltaP variables to match the deltaP variable values,
    or vice versa (depending on which is fixed).

    This is generalised so you can pass any component name.
    """
    inverted_component_name = f"{component_name}_inverted"
    component = getattr(block, component_name)
    inverted_component = getattr(block, inverted_component_name)

    for indexes in component.index_set():
        if component[indexes].fixed:
            # we can savely assume that the inverted variable is not
            # fixed, otherwise this would be an overconstrained model.
            inverted_component[indexes].value = -component[indexes].value
        elif inverted_component[indexes].fixed:
            component[indexes].value = -inverted_component[indexes].value
        else:
            # neither variable is fixed,
            # so let's just leave them as is.
            # Hopefully the solver can figure it out.
            pass

def disable_inverted(block: Block, component_name: str) -> bool:
    """
    Disables the inverted variables and constraints for a given component.
    Returns True if the inverted variable was fixed, false otherwise.
    """
    inverted_component_name = f"{component_name}_inverted"
    constraint_name = f"{inverted_component_name}_constraint"

    inverted_constraint = getattr(block, constraint_name)

    inverted_constraint.deactivate()


def enable_inverted(block: Block, component_name: str):
    """
    Enables the inverted variables and constraints for a given component.
    """
    inverted_component_name = f"{component_name}_inverted"
    constraint_name = f"{inverted_component_name}_constraint"

    inverted_constraint = getattr(block, constraint_name)

    inverted_constraint.activate()