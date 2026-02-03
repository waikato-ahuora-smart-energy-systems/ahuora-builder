import pyomo.environ as pe# Pyomo environment
from idaes.core import FlowsheetBlock, StateBlock
from idaes.models.unit_models import HeatExchanger
from idaes.models.properties.general_helmholtz import (
    HelmholtzParameterBlock,
    PhaseType,
    StateVars,
)
from idaes.models.unit_models.heat_exchanger import HX0DInitializer
from idaes.models.unit_models.heat_exchanger import delta_temperature_lmtd_callback
from idaes.models.properties import iapws95
from ahuora_builder.custom.SimpleEffectivenessHX_DH  import HeatExchangerEffectiveness  
# from milk_props_full_config  import MilkParameterBlock  
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
from idaes.core.util import DiagnosticsToolbox
from multiprocessing import freeze_support
from property_packages.build_package import build_package

#--------------------------------------------------------------
#Only works with smooth temperature need the Ahuora smooth helmhotlz, the Ahuora smooth Helmholtz does not work on windows
#--------------------------------------------------------------

if __name__ == '__main__':
    freeze_support()
    # Create an empty flowsheet and steam property parameter block.
    model = pe.ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=False)
    model.fs.properties = build_package("helmholtz",["water"],["Liq","Vap"])  #HelmholtzParameterBlock(pure_component="h2o",state_vars=StateVars.PH,phase_presentation=PhaseType.MIX)
    dt = DiagnosticsToolbox(model)
    # Add a Heater model to the flowsheet.
    model.fs.heat_exchanger = HeatExchangerEffectiveness(
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": model.fs.properties},
        tube={"property_package": model.fs.properties},

    )

    model.fs.heat_exchanger.effectiveness.fix(1)
    model.fs.heat_exchanger.shell_inlet.flow_mol.fix(10)
    P_hot = 270280
    T_hot = 200 + 273
    h = model.fs.properties.htpx(T=T_hot*pe.units.K, p= P_hot*pe.units.Pa)
    print(h)
    model.fs.heat_exchanger.shell_inlet.pressure.fix(P_hot)
    model.fs.heat_exchanger.shell_inlet.enth_mol.fix(h)

    P_cold = 101325
    T_cold = 50 + 273
    model.fs.heat_exchanger.tube_inlet.flow_mol.fix(5)
    model.fs.heat_exchanger.tube_inlet.pressure.fix(P_cold)

    h = model.fs.properties.htpx(T=T_cold*pe.units.K, p= P_cold*pe.units.Pa)
    print(h)
    model.fs.heat_exchanger.tube_inlet.enth_mol.fix(h)
    # Perform degrees of freedom analysis
    dof = degrees_of_freedom(model)
    print("Degrees of Freedom:", dof)


    # Initialize the model
    # initializer = HX0DInitializer()
    # initializer.initialize(model.fs.heat_exchanger)
    model.fs.heat_exchanger.initialize(outlvl=idaeslog.INFO_HIGH)
    model.fs.heat_exchanger.report()

    # Solve the models
    solver = pe.SolverFactory('ipopt')
    results = solver.solve(model, tee=True)

    # Display the results
    model.fs.heat_exchanger.report()
    print("Hot inlet enthalpy theory: ", model.fs.properties.htpx(p= P_hot*pe.units.Pa, T= T_hot*pe.units.K), "J/mol")
    print("Cold inlet enthalpy theory: ", model.fs.properties.htpx(p= P_cold*pe.units.Pa, T= T_cold*pe.units.K), "J/mol")
    print("Hot inlet enthalpy: ", pe.value(model.fs.heat_exchanger.hot_side.properties_in[0].enth_mol), "J/mol")
    print("Cold inlet enthalpy: ", pe.value(model.fs.heat_exchanger.cold_side.properties_in[0].enth_mol), "J/mol")
    print("Hot outlet enthalpy_max: ", pe.value(model.fs.heat_exchanger.properties_hotside[0].enth_mol), "J/mol")
    print("Cold outlet enthalpy_max: ", pe.value(model.fs.heat_exchanger.properties_coldside[0].enth_mol), "J/mol")
    print("dH_hot: ", pe.value(model.fs.heat_exchanger.delta_h_hot_qmax[0]), "J/mol")
    print("dH_cold: ", pe.value(model.fs.heat_exchanger.delta_h_cold_qmax[0]), "J/mol")
    print("Duty: ", pe.value(model.fs.heat_exchanger.heat_duty[0])/1e3, "kW")
    print("Qmax: ", pe.value(model.fs.heat_exchanger.Qmax[0])/1e3, "kW")


    eff = pe.value(model.fs.heat_exchanger.effectiveness[0])
    eff_apparent = pe.value(model.fs.heat_exchanger.heat_duty[0]/model.fs.heat_exchanger.Qmax[0])
    print("Effectiveness set: ", eff)
    print("Effectiveness apparent: ", eff_apparent)


