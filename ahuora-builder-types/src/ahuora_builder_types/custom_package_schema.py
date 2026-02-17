from pydantic import BaseModel, Field


class CustomCompoundPropertiesSchema(BaseModel):
    cp_mol_ig_comp_coeff_A : float
    cp_mol_ig_comp_coeff_B : float
    cp_mol_ig_comp_coeff_C : float
    cp_mol_ig_comp_coeff_D : float
    omega : float
    mw : float
    pressure_crit : float
    temperature_crit : float
    enth_mol_form_vap_comp_ref : float
    entr_mol_form_vap_comp_ref : float
    pressure_sat_comp_coeff_A : float
    pressure_sat_comp_coeff_B : float
    pressure_sat_comp_coeff_C : float
    pressure_sat_comp_coeff_D : float

class CustomPropertyPackagePropertiesSchema(BaseModel):
    flow_mol_min : float
    flow_mol_nominal : float
    flow_mol_max : float
    temperature_min : float
    temperature_nominal : float
    temperature_max : float
    pressure_min : float
    pressure_nominal : float
    pressure_max : float
    pressure_ref : float
    temperature_ref : float
