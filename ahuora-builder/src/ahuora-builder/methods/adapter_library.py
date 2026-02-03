from idaes.models.unit_models import (
    Pump,
    Heater,
    Mixer,
    StateJunction,
    HeatExchanger,
    Turbine,
    PressureChanger,
)
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.separator import SplittingType, EnergySplittingType
from ahuora_builder.custom.heat_exchanger_1d_wrapper import HeatExchanger1DWrapper
from . import adapter_methods as methods
from ..custom.custom_heat_exchanger import CustomHeatExchanger
from ..custom.PySMOModel import PySMOModel
from ..custom.energy.link import Link
from ..custom.energy.power_property_package import PowerParameterBlock
from ..custom.energy.ac_property_package import acParameterBlock
from ..custom.energy.transformer_property_package import transformerParameterBlock
from ..custom.updated_pressure_changer import Compressor
from ..custom.energy.solar import Solar
from ..custom.custom_valve import Valve
from ..custom.energy.wind import Wind
from ..custom.energy.energy_mixer import EnergyMixer
from ..custom.energy.bus    import Bus
from ..custom.energy.acBus    import acBus
from ..custom.energy.transmissionLine import transmissionLine
from ..custom.valve_wrapper import ValveWrapper
from ..custom.energy.load import Load
from ..custom.energy.hydro import Hydro
from ..custom.energy.transformer import Transformer
from ..custom.energy.storage import Storage
from ..custom.direct_steam_injection import Dsi
from ..custom.translator import GenericTranslator
from idaes.models.control.controller import  ControllerType, ControllerMVBoundType
from ..custom.PIDController import PIDController2
from ..custom.energy.grid import Grid
from ..custom.custom_heater import DynamicHeater
from ..custom.custom_tank import CustomTank
from ..custom.custom_compressor import CustomCompressor
from ..custom.custom_pump import CustomPump
from ..custom.integration_block import IntegrationBlock
from idaes.models.unit_models.cstr import CSTR
from ..custom.energy.energy_splitter import EnergySplitter
from ..custom.custom_turbine import CustomTurbine
from ..custom.energy.mainDistributionBoard import MDB
from ..custom.custom_separator import CustomSeparator
from ..custom.custom_variable import CustomVariable
from ..custom.thermal_utility_systems.steam_header import SteamHeader
from ..custom.thermal_utility_systems.header import simple_header
from ..custom.reactions.hda_stoich import HDAStoichiometricReactor
from ..custom.custom_heat_exchanger_1d import CustomHeatExchanger1D
from ..custom.thermal_utility_systems.simple_heat_pump import SimpleHeatPump
from ..custom.thermal_utility_systems.steam_user import SteamUser
from ..custom.thermal_utility_systems.desuperheater import Desuperheater
from idaes.models_extra.power_generation.unit_models.waterpipe import WaterPipe
from ..custom.SimpleEffectivenessHX_DH import HeatExchangerEffectiveness
from idaes.models.unit_models.heat_exchanger_lc import HeatExchangerLumpedCapacitance
from idaes.models_extra.column_models.plate_heat_exchanger import PlateHeatExchanger
from ..custom.custom_cooler import CustomCooler

class UnitModelConstructor:
    """
    Schema for adapter library items
    """

    def __init__(self, model_constructor: type, arg_parsers: dict):
        # idaes model constructor
        self.model_constructor: type = model_constructor
        # dictionary of argument parsers
        self.arg_parsers: dict[str,methods.AdapterBase] = arg_parsers



"""
A dictionary of adapters for the idaes model constructors
"""
AdapterLibrary: dict[str, UnitModelConstructor] = {
        "pump": UnitModelConstructor(
            CustomPump,
            {
                "property_package": methods.PropertyPackage(),
                "dynamic": methods.Constant(False),
                "power_property_package": methods.PowerPropertyPackage(),

            },
        ),
        "heater": UnitModelConstructor(
            DynamicHeater,
            {
                "property_package": methods.PropertyPackage(),
                "has_pressure_change": methods.Value(),
                "dynamic": methods.Value(),
                "has_holdup": methods.Value(),
            },
        ),
        "compressor": UnitModelConstructor(
            CustomCompressor,
            {
                "property_package": methods.PropertyPackage(),
                "power_property_package": methods.PowerPropertyPackage(),
                "dynamic": methods.Constant(False),
            },
        ),
        "turbine": UnitModelConstructor(
            CustomTurbine,
            {
                "property_package": methods.PropertyPackage(),
                "power_property_package": methods.PowerPropertyPackage(),
                "dynamic": methods.Constant(False),
                "calculation_method": methods.Constant("isentropic"),
            },
        ),
        "willans_turbine": UnitModelConstructor(
            CustomTurbine,
            {
                "property_package": methods.PropertyPackage(),
                "power_property_package": methods.PowerPropertyPackage(),
                "dynamic": methods.Constant(False),
                "calculation_method": methods.Constant("simple_willans"),
            },
        ),
        "pl_turbine": UnitModelConstructor(
            CustomTurbine,
            {
            "property_package": methods.PropertyPackage(),
            "power_property_package": methods.PowerPropertyPackage(),
            "dynamic": methods.Constant(False),
            "calculation_method": methods.Constant("part_load_willans"),  
            }
        ),
        "dts_turbine": UnitModelConstructor(
            CustomTurbine,
            {
            "property_package": methods.PropertyPackage(),
            "power_property_package": methods.PowerPropertyPackage(),
            "dynamic": methods.Constant(False),
            "calculation_method": methods.Constant("Tsat_willans"),  
            }
        ),
        "bs_turbine": UnitModelConstructor(
            CustomTurbine,
            {
            "property_package": methods.PropertyPackage(),
            "power_property_package": methods.PowerPropertyPackage(),
            "dynamic": methods.Constant(False),
            "calculation_method": methods.Constant("BPST_willans"),  
            }
        ),
        "cs_turbine": UnitModelConstructor(
            CustomTurbine,
            {
            "property_package": methods.PropertyPackage(),
            "power_property_package": methods.PowerPropertyPackage(),
            "dynamic": methods.Constant(False),
            "calculation_method": methods.Constant("CT_willans"),  
            }
         ),  
        "mixer": UnitModelConstructor(
            Mixer,
            {
                "property_package": methods.PropertyPackage(),
                "num_inlets": methods.Value(),
            },
        ),
        "Tank": UnitModelConstructor(
            CustomTank,
            {
                "property_package": methods.PropertyPackage(),
                "dynamic": methods.Value(),
                "has_holdup": methods.Value(),
                "tank_type": methods.Constant("vertical_cylindrical_tank"),
                "has_heat_transfer": methods.Constant(True),
            }
        ),
        "heatExchanger": UnitModelConstructor(
            CustomHeatExchanger,
            {
                "delta_temperature_callback": methods.Constant(
                    delta_temperature_underwood_callback
                ),
                "hot_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(True),
                    }
                ),
                "cold_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(True),
                    }
                ),
                "dynamic": methods.Constant(False),
            },
        ),
        "heatPump": UnitModelConstructor(
            SimpleHeatPump,
            {
                "source": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(False),
                    }
                ),
                "sink": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(False),
                    }
                ),
                "dynamic": methods.Constant(False),
            },
        ),
        "valve": UnitModelConstructor(
            ValveWrapper, # type: ignore
            {
                "property_package": methods.PropertyPackage(),
                "enable_coefficients": methods.Value(),
                "valve_function": methods.Value(),
                "dynamic": methods.Constant(False),
            },
        ),
        "cooler": UnitModelConstructor(
            CustomCooler,
            {
                "property_package": methods.PropertyPackage(),
                "has_pressure_change": methods.Value(),
                "dynamic": methods.Constant(False),
            },
        ),
        "splitter": UnitModelConstructor(
            CustomSeparator,
            {
                "property_package": methods.PropertyPackage(),
                "num_outlets": methods.Value(),
            },
        ),
        "phaseSeparator": UnitModelConstructor(
            CustomSeparator,
            {
                "property_package": methods.PropertyPackage(),
                "num_outlets": methods.Value(),
                "split_basis": methods.Constant(
                    SplittingType.phaseFlow
                ),
                "energy_split_basis": methods.Constant(
                    EnergySplittingType.enthalpy_split
                ),
            },
        ),
        "compoundSeparator": UnitModelConstructor(
            CustomSeparator,
            {
                "property_package": methods.PropertyPackage(),
                "num_outlets": methods.Value(),
                "split_basis": methods.Constant(
                    SplittingType.componentFlow
                ),
            },
        ),
        "machineLearningBlock": UnitModelConstructor(PySMOModel, {
            "property_package": methods.PropertyPackage(),
            "model": methods.Value(),
            "ids": methods.Value(),
            "unitopNames": methods.Value(),
            "num_inlets": methods.Value(),
            "num_outlets": methods.Value(),
        }),
        "link": UnitModelConstructor(
            Link,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
        "solar": UnitModelConstructor(
            Solar,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
        "wind": UnitModelConstructor(
            Wind,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
        "energy_mixer": UnitModelConstructor(
            EnergyMixer,
            {
                "property_package": methods.PowerPropertyPackage(),
                "num_inlets": methods.Value(),
            },
        ),
        "bus": UnitModelConstructor(
            Bus,
            {
                "property_package": methods.PowerPropertyPackage(),
                "num_inlets": methods.Value(),
                
            },
         ), 
        "acBus": UnitModelConstructor(
            acBus,
            {
                "property_package": methods.acPropertyPackage(),
                "num_inlets": methods.Value(),
                "num_outlets": methods.Value(),
            },       
        ),
        "transmissionLine": UnitModelConstructor(
            transmissionLine,
            {
                "property_package": methods.PowerPropertyPackage(),
                "num_inlets": methods.Value(),
                "num_outlets": methods.Value(),
            },
        ),
        "load": UnitModelConstructor(
            Load,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
        "hydro": UnitModelConstructor(
            Hydro,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
    
        "transformer": UnitModelConstructor(
            Transformer,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
        "storage": UnitModelConstructor(
            Storage,
            {
                "property_package": methods.PowerPropertyPackage(),
                "has_holdup": methods.Value(),
                "dynamic": methods.Value(),
            },
        ),
        "direct_steam_injection": UnitModelConstructor(
            Dsi,
            {
                "property_package": methods.PropertyPackage(),
                "steam_property_package": methods.PropertyPackage(),
            },
        ),
        "translator": UnitModelConstructor(
            GenericTranslator,
            {
                "inlet_property_package": methods.PropertyPackage(),
                "outlet_property_package": methods.PropertyPackage(),
                "translator_type": methods.Value(),
            },
        ),
        "pid_controller": UnitModelConstructor(
            PIDController2,
            {
                "dynamic": methods.Constant(True),
                "controller_type": methods.Constant(ControllerType.PI),
                "calculate_initial_integral": methods.Constant(True),
                "mv_bound_type": methods.Constant(ControllerMVBoundType.SMOOTH_BOUND),
            },
        ),
        "custom_variable": UnitModelConstructor(
            CustomVariable,
            {},
        ),
        "grid": UnitModelConstructor(
            Grid,
            {
                "property_package": methods.PowerPropertyPackage(),
            },
        ),
        "integration": UnitModelConstructor(
            IntegrationBlock,
            {
            },
        ),
        "stoich_hda": UnitModelConstructor(
            HDAStoichiometricReactor,
            {
                "property_package": methods.ReactorPropertyPackage(),
                "has_heat_transfer": methods.Constant(True),
                #"has_heat_of_reaction": methods.Value(),
                #"has_pressure_change": methods.Value(),
                "has_holdup": methods.Constant(False),
                "dynamic": methods.Constant(False),
                "reaction_package": methods.ReactionPackage(),
            }, 
        ),
        "stoich_hda": UnitModelConstructor(
            HDAStoichiometricReactor,
            {
                "property_package": methods.ReactorPropertyPackage(),
                "has_heat_transfer": methods.Constant(True),
                #"has_heat_of_reaction": methods.Value(),
                #"has_pressure_change": methods.Value(),
                "has_holdup": methods.Constant(False),
                "dynamic": methods.Constant(False),
                "reaction_package": methods.ReactionPackage(),
            }, 
        ),
        "RCT_CSTR": UnitModelConstructor(
            CSTR,
            {
                "property_package": methods.ReactorPropertyPackage(),
                "has_heat_transfer": methods.Value(),
                "has_heat_of_reaction": methods.Value(),
                "has_pressure_change": methods.Value(),
                "has_holdup": methods.Value(),
                "dynamic": methods.Value(),
                "reaction_package": methods.ReactionPackage(),
            },
            
        ),
        "energy_splitter": UnitModelConstructor(
            EnergySplitter,
            {
                "property_package": methods.PowerPropertyPackage(),
                "num_inlets": methods.Value(),
                "num_outlets": methods.Value(),
            },
        ),
        "mdb": UnitModelConstructor(
            MDB,
            {
                "property_package": methods.PowerPropertyPackage(),
                "num_inlets": methods.Value(),
                "num_outlets": methods.Value(),
            }
        ),
        "header": UnitModelConstructor(
            SteamHeader,
            {
                "property_package": methods.PropertyPackage(),
                "num_inlets": methods.Value(),
                "num_outlets": methods.Value(),
            },
        ),
        "simple_header": UnitModelConstructor(
            simple_header,
            {
                "property_package": methods.PropertyPackage(),
                "num_inlets": methods.Value(),
                "num_outlets": methods.Value(),
            },
        ),
        "heat_exchanger_1d": UnitModelConstructor(
            HeatExchanger1DWrapper,
            {
                "hot_side": methods.Dictionary({
                    "property_package": methods.PropertyPackage(),
                    "transformation_method": methods.Value(),
                    "transformation_scheme": methods.Value(),
                    "has_pressure_change": methods.Constant(False),
                }),
                "cold_side": methods.Dictionary({
                    "property_package": methods.PropertyPackage(),
                    "transformation_method": methods.Value(),
                    "transformation_scheme": methods.Value(),
                    "has_pressure_change": methods.Constant(False),
                }),
                "finite_elements": methods.Value(),
                "collocation_points": methods.Value(),
                "flow_type": methods.Value(),
            },
        ),
        "steam_user": UnitModelConstructor(
            SteamUser,
            {
                "property_package": methods.PropertyPackage(),
            }
        ),
         "desuperheater": UnitModelConstructor(
            Desuperheater,
            {
                "property_package": methods.PropertyPackage(),
            }
        ),
        "heat_exchanger_ntu": UnitModelConstructor(
            HeatExchangerEffectiveness,
            {
                "hot_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(True),
                    }
                ),
                "cold_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(True),
                    }
                ),
                "dynamic": methods.Constant(False),
            },
        ),
        "waterpipe": UnitModelConstructor(
            WaterPipe,
            {
                "property_package": methods.PropertyPackage(),
                "has_heat_transfer": methods.Constant(False),
                "has_pressure_change": methods.Constant(True),
            }
        ),
        "heat_exchanger_lc": UnitModelConstructor(
            HeatExchangerLumpedCapacitance,
            {
                "hot_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(False),
                    }
                ),
                "cold_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(False),
                    }
                ),
                "dynamic_heat_balance": methods.Constant(False),
            },
        ),
        "plate_heat_exchanger": UnitModelConstructor(
            PlateHeatExchanger,
            {
                "hot_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(True),
                    }
                ),
                "cold_side": methods.Dictionary(
                    {
                        "property_package": methods.PropertyPackage(),
                        "has_pressure_change": methods.Constant(True),
                    }
                ),
                "dynamic": methods.Constant(False),
            },
        ),
    }
