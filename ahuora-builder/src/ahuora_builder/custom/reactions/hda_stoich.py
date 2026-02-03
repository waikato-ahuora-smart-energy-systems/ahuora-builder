from idaes.core import declare_process_block_class
from idaes.models.unit_models.stoichiometric_reactor import StoichiometricReactorData
from pyomo.environ import Var, Reference

@declare_process_block_class("HDAStoichiometricReactor")
class HDAStoichiometricReactorData(StoichiometricReactorData):
    def build(self,*args, **kwargs):
        super().build(*args, **kwargs)
        # reaction id in https://github.com/IDAES/examples/blob/50065c4cc4de96a8dc1cad833101df2d96574ac4/idaes_examples/mod/hda/hda_reaction.py#L4
        self.hda_extent = Reference(self.rate_reaction_extent[:,"R1"])