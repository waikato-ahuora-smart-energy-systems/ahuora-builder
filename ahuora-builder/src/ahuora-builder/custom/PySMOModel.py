# Methods and helper functions to train and use a surrogate valve.
from pyomo.core.base.expression import ScalarExpression
from idaes.core import MaterialBalanceType, ControlVolume0DBlock, declare_process_block_class, EnergyBalanceType, MomentumBalanceType, MaterialBalanceType, useDefault, UnitModelBlockData
from pyomo.common.config import ConfigBlock, ConfigValue, In
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.util.config import is_physical_parameter_block
from ahuora_builder.methods.expression_parsing import get_property_from_id
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from io import StringIO
import json
from pyomo.environ import Var, Constraint
from pyomo.environ import (
    Constraint,
    Set,
    Var,
    Suffix,
    units as pyunits,
)

def make_control_volume(unit, name, config):
    if config.dynamic is not False:
        raise ValueError('SurrogateValve does not support dynamics')
    if config.has_holdup is not False:
        raise ValueError('SurrogateValve does not support holdup')

    control_volume = ControlVolume0DBlock(
                                          dynamic=config.dynamic,
                                          has_holdup=config.has_holdup,
                                          property_package=config.property_package,
                                          property_package_args=config.property_package_args)

    # Add the control volume block to the unit
    setattr(unit, name, control_volume)

    control_volume.add_state_blocks(has_phase_equilibrium=config.has_phase_equilibrium)
    # control_volume.add_material_balances(balance_type=config.material_balance_type,
    #                                      has_phase_equilibrium=config.has_phase_equilibrium)
    # control_volume.add_total_enthalpy_balances(has_heat_of_reaction=False, 
    #                                            has_heat_transfer=False, 
    #                                            has_work_transfer=False)
@declare_process_block_class("PySMOModel")
class PySMOModelData(UnitModelBlockData):
    CONFIG = UnitModelBlockData.CONFIG()
    # Declare all the standard config arguments for the control_volume
    CONFIG.declare("material_balance_type", ConfigValue(default=MaterialBalanceType.componentPhase, domain=In(MaterialBalanceType)))
    CONFIG.declare("energy_balance_type", ConfigValue(default=EnergyBalanceType.enthalpyTotal, domain=In([EnergyBalanceType.enthalpyTotal])))
    CONFIG.declare("momentum_balance_type", ConfigValue(default=MomentumBalanceType.none, domain=In([MomentumBalanceType.none])))
    CONFIG.declare("has_phase_equilibrium", ConfigValue(default=False, domain=In([False])))
    CONFIG.declare("has_pressure_change", ConfigValue(default=False, domain=In([False])))
    CONFIG.declare("property_package", ConfigValue(default=useDefault, domain=is_physical_parameter_block))
    CONFIG.declare("property_package_args", ConfigBlock(implicit=True))
    # no other args need to be declared, we are just hardcoding the valve model.
    CONFIG.declare("model", ConfigValue())
    CONFIG.declare("ids", ConfigValue())
    CONFIG.declare("unitopNames", ConfigValue())
    CONFIG.declare(
            "num_inlets",
            ConfigValue(
                default=False,
                domain=int,
                description="Number of inlets to add",
                doc="Number of inlets to add",
            ),
        )
    CONFIG.declare(
        "num_outlets",
        ConfigValue(
            default=False,
            domain=int,
            description="Number of outlets to add",
            doc="Number of outlets to add",
        ),
    )
    
    def build(self):
        super(PySMOModelData, self).build()
        self.CONFIG.dynamic = False
        self.CONFIG.has_holdup = False
        # This function handles adding the control volume block to the unit,
        # and addiing the necessary material and energy balances.
        make_control_volume(self, "control_volume", self.CONFIG)

        # self.add_inlet_port()
        # self.add_outlet_port()

        # Defining parameters of state block class
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True  # inlet block is an inlet

        # Add state blocks for inlet, outlet, and waste
        # These include the state variables and any other properties on demand
        num_inlets = self.config.num_inlets
        self.inlet_list = [ "inlet_" + str(i+1) for i in range(num_inlets) ]    
        self.inlet_set = Set(initialize=self.inlet_list) 
        self.inlet_blocks = []

        for name in self.inlet_list:
            # add properties_inlet_1, properties_inlet2 etc
            state_block = self.config.property_package.state_block_class(
                self.flowsheet().config.time, doc="inlet ml", **tmp_dict
            )
            self.inlet_blocks.append(state_block)
            # Dynamic equivalent to self.properties_inlet_1 = stateblock
            setattr(self,"properties_" + name, state_block)
            # also add the port
            self.add_port(name=name,block=state_block)


        # Add outlet state blocks

        num_outlets = self.config.num_outlets
        self.outlet_list = [ "outlet_" + str(i+1) for i in range(num_outlets) ]
        self.outlet_set = Set(initialize=self.outlet_list)
        self.outlet_blocks = []

        for name in self.outlet_list:
            tmp_dict["defined_state"] = False
            state_block = self.config.property_package.state_block_class(
                self.flowsheet().config.time, doc="outlet ml", **tmp_dict
            )
            self.outlet_blocks.append(state_block)
            setattr(self,"properties_" + name, state_block)
            self.add_port(name=name,block=state_block)
        
        
        
        # Add variables for custom properties        
        names = self.config.unitopNames
        for name in names:
            self.add_component(
                name, Var(self.flowsheet().time, initialize=10)
            )
    
    def initialize(self, *args, **kwargs):
        model_data = self.config.model
        json_str = json.dumps(model_data)
        f = StringIO(json_str)

        model = PysmoSurrogate.load(f)
        ids = self.config.ids
        fs = self.flowsheet()
        # TODO: Make surrogate models work with dynamics. This involves making a surrogate model for each time step.
        # Not sure if idaes has a framework for doing this or not.
        # For now, we are just assuming steady state and time_index=0.
        inputs = [get_property_from_id(fs, i,0) for i in ids["input"]]
        outputs = [get_property_from_id(fs, i,0) for i in ids["output"]]

        self.check_is_expression(inputs)
        self.check_is_expression(outputs)
        
        self.surrogate = SurrogateBlock(concrete=True)
        self.surrogate.build_model(model,input_vars=inputs, output_vars=outputs)
        
    def check_is_expression(self, vars):
        for index, var in enumerate(vars):
            if isinstance(var, ScalarExpression):
                name = f"{var.name}_{index}"
                new_var = Var(self.flowsheet().time, initialize=1)
                self.add_component(name, new_var)
                def constraint_rule(model, t):
                    if var.is_indexed():
                        return new_var[t] == var[t]
                    else:
                        return new_var[t] == var

                self.add_component(f"{name}_constraint", Constraint(self.flowsheet().time, rule=constraint_rule))
                vars[index] = new_var

    def _get_stream_table_contents(self, time_point=0):

        io_dict = {}
        for inlet_name in self.inlet_list:
            io_dict[inlet_name] = getattr(self, inlet_name) # get a reference to the port
        
        out_dict = {}
        for outlet_name in self.outlet_list:
            out_dict[outlet_name] = getattr(self, outlet_name) # get a reference to the port
