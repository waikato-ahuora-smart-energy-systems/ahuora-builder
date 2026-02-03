from .custom_heat_exchanger_1d import CustomHeatExchanger1D
from idaes.models.unit_models.heat_exchanger import HeatExchangerFlowPattern

# Map frontend string values to IDAES enums
FLOW_TYPE_MAP = {
    "countercurrent": HeatExchangerFlowPattern.countercurrent,
    "cocurrent": HeatExchangerFlowPattern.cocurrent,
}

def HeatExchanger1DWrapper(**kwargs):
    """
    Wrapper for CustomHeatExchanger1D that handles string-to-enum conversion
    and provides default values for required parameters, defaulting to collocation.
    """
    # Convert flow_type string to enum
    flow_type = kwargs.pop('flow_type', None)
    if flow_type is None:
        flow_type = "countercurrent"
    
    if isinstance(flow_type, str):
        if flow_type.lower() in FLOW_TYPE_MAP:
            flow_type_enum = FLOW_TYPE_MAP[flow_type.lower()]
        else:
            raise ValueError(f"Unknown flow_type: {flow_type}. Must be one of: {list(FLOW_TYPE_MAP.keys())}")
    else:
        flow_type_enum = flow_type
    
    kwargs['flow_type'] = flow_type_enum
    
    # Return the heat exchanger with fixed parameters
    return CustomHeatExchanger1D(**kwargs)