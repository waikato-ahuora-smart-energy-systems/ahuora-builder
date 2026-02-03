from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent



def is_scalar_reference(component: IndexedComponent) -> bool:
    # The only key in it should be None
    return list(component) == [None]