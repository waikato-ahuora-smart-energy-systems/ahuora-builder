from idaes.models.unit_models.separator import SeparatorData, SplittingType

from functools import partial
from pandas import DataFrame

from pyomo.environ import (
    Block,
    check_optimal_termination,
    Constraint,
    Param,
    Reals,
    Reference,
    Set,
    Var,
    value,
)
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In, ListOf, Bool

from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    MaterialBalanceType,
    MomentumBalanceType,
    MaterialFlowBasis,
    VarLikeExpression,
)
from idaes.core.util.config import (
    is_physical_parameter_block,
    is_state_block,
)
from idaes.core.util.exceptions import (
    BurntToast,
    ConfigurationError,
    PropertyNotSupportedError,
    InitializationError,
)
from idaes.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.units_of_measurement import report_quantity
from idaes.core.initialization import ModularInitializerBase


# This only changes a couple of lines in the original SeparatorData class, to not fix state variables by default.
# The state block initialisation already does this if needed, so we can just set their value.
# This is because if the state block has extra constraints, such as for flow_mass, then fixing flow_mol will over-define the system.
# It might be worth making this a pr to idaes.

@declare_process_block_class("CustomSeparator")
class CustomSeparatorData(SeparatorData):

    def initialize_build(
        blk, outlvl=idaeslog.NOTSET, optarg=None, solver=None, hold_state=False
    ):
        """
        Initialization routine for separator

        Keyword Arguments:
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)
            hold_state : flag indicating whether the initialization routine
                     should unfix any state variables fixed during
                     initialization, **default** - False. **Valid values:**
                     **True** - states variables are not unfixed, and a dict of
                     returned containing flags for which states were fixed
                     during initialization, **False** - state variables are
                     unfixed after initialization by calling the release_state
                     method.

        Returns:
            If hold_states is True, returns a dict containing flags for which
            states were fixed during initialization.
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        # Initialize mixed state block
        if blk.config.mixed_state_block is not None:
            mblock = blk.config.mixed_state_block
        else:
            mblock = blk.mixed_state
        flags = mblock.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            hold_state=True,
        )

        # Solve for split fractions only
        component_status = {}
        for c in blk.component_objects((Block, Constraint)):
            for i in c:
                if not c[i].local_name == "sum_split_frac":
                    # Record current status of components to restore later
                    component_status[c[i]] = c[i].active
                    c[i].deactivate()

        if degrees_of_freedom(blk) != 0:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(blk, tee=slc.tee)
                init_log.info(
                    "Initialization Step 1 Complete: {}".format(idaeslog.condition(res))
                )

        for c, s in component_status.items():
            if s:
                c.activate()

        if blk.config.ideal_separation:
            # If using ideal splitting, initialization should be complete
            return flags

        # Initialize outlet StateBlocks
        outlet_list = blk.create_outlet_list()

        # Premises for initializing outlet states:
        # 1. Intensive states remain unchanged - this is either a valid premise
        # or the actual state is impossible to calculate without solving the
        # full separator model.
        # 2. Extensive states are use split fractions if index matches, or
        # average of split fractions for outlet otherwise
        for o in outlet_list:
            # Get corresponding outlet StateBlock
            o_block = getattr(blk, o + "_state")

            # Create dict to store fixed status of state variables
            o_flags = {}
            for t in blk.flowsheet().time:

                # Calculate values for state variables
                s_vars = o_block[t].define_state_vars()
                for v in s_vars:
                    for k in s_vars[v]:
                        # Record whether variable was fixed or not
                        o_flags[t, v, k] = s_vars[v][k].fixed

                        # If fixed, use current value
                        # otherwise calculate guess from mixed state and fix
                        if not s_vars[v][k].fixed:
                            m_var = getattr(mblock[t], s_vars[v].local_name)
                            if "flow" in v:
                                # If a "flow" variable, is extensive
                                # Apply split fraction
                                if blk.config.split_basis == SplittingType.totalFlow:
                                    # All flows split by outlet
                                    s_vars[v][k].set_value(
                                        value(m_var[k] * blk.split_fraction[(t, o)])
                                    )
                                elif "_phase_comp" in v:
                                    # Need to match indices, but use split frac
                                    if (
                                        blk.config.split_basis
                                        == SplittingType.phaseComponentFlow
                                    ):
                                        s_vars[v][k].set_value(
                                            value(
                                                m_var[k]
                                                * blk.split_fraction[(t, o) + (k,)]
                                            )
                                        )
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.phaseFlow
                                    ):
                                        s_vars[v][k].set_value(
                                            value(
                                                m_var[k]
                                                * blk.split_fraction[(t, o) + (k[0],)]
                                            )
                                        )
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.componentFlow
                                    ):
                                        s_vars[v][k].set_value(
                                            value(
                                                m_var[k]
                                                * blk.split_fraction[(t, o) + (k[1],)]
                                            )
                                        )
                                    else:
                                        raise BurntToast(
                                            "{} encountered unrecognised "
                                            "SplittingType. This should not "
                                            "occur - please send this bug to "
                                            "the IDAES developers.".format(blk.name)
                                        )
                                elif "_phase" in v:
                                    if (
                                        blk.config.split_basis
                                        == SplittingType.phaseComponentFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, k, j]
                                                for j in mblock.component_list
                                            )
                                            / len(mblock.component_list)
                                        )
                                        s_vars[v][k].set_value(value(m_var[k] * avg_split))
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.phaseFlow
                                    ):
                                        s_vars[v][k].set_value(
                                            value(
                                                m_var[k]
                                                * blk.split_fraction[(t, o) + (k,)]
                                            )
                                        )
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.componentFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, j]
                                                for j in mblock.component_list
                                            )
                                            / len(mblock.component_list)
                                        )
                                        s_vars[v][k].set_value(value(m_var[k] * avg_split))
                                    else:
                                        raise BurntToast(
                                            "{} encountered unrecognised "
                                            "SplittingType. This should not "
                                            "occur - please send this bug to "
                                            "the IDAES developers.".format(blk.name)
                                        )
                                elif "_comp" in v:
                                    if (
                                        blk.config.split_basis
                                        == SplittingType.phaseComponentFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, p, k]
                                                for p in mblock.phase_list
                                            )
                                            / len(mblock.phase_list)
                                        )
                                        s_vars[v][k].set_value(value(m_var[k] * avg_split))
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.phaseFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, p]
                                                for p in mblock.phase_list
                                            )
                                            / len(mblock.phase_list)
                                        )
                                        s_vars[v][k].set_value(value(m_var[k] * avg_split))
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.componentFlow
                                    ):
                                        s_vars[v][k].set_value(
                                            value(
                                                m_var[k]
                                                * blk.split_fraction[(t, o) + (k,)]
                                            )
                                        )
                                    else:
                                        raise BurntToast(
                                            "{} encountered unrecognised "
                                            "SplittingType. This should not "
                                            "occur - please send this bug to "
                                            "the IDAES developers.".format(blk.name)
                                        )
                                else:
                                    # Assume unindexed extensive state
                                    # Need average split
                                    if (
                                        blk.config.split_basis
                                        == SplittingType.phaseComponentFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, p, j]
                                                for (p, j) in mblock.phase_component_set
                                            )
                                            / len(mblock.phase_component_set)
                                        )
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.phaseFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, p]
                                                for p in mblock.phase_list
                                            )
                                            / len(mblock.phase_list)
                                        )
                                    elif (
                                        blk.config.split_basis
                                        == SplittingType.componentFlow
                                    ):
                                        # Need average split fraction
                                        avg_split = value(
                                            sum(
                                                blk.split_fraction[t, o, j]
                                                for j in mblock.component_list
                                            )
                                            / len(mblock.component_list)
                                        )
                                    else:
                                        raise BurntToast(
                                            "{} encountered unrecognised "
                                            "SplittingType. This should not "
                                            "occur - please send this bug to "
                                            "the IDAES developers.".format(blk.name)
                                        )
                                    s_vars[v][k].set_value(value(m_var[k] * avg_split))
                            else:
                                # Otherwise intensive, equate to mixed stream
                                s_vars[v][k].set_value(m_var[k].value)

            # Call initialization routine for outlet StateBlock
            o_block.initialize(
                outlvl=outlvl,
                optarg=optarg,
                solver=solver,
                hold_state=False,
            )

            # Revert fixed status of variables to what they were before
            for t in blk.flowsheet().time:
                s_vars = o_block[t].define_state_vars()
                for v in s_vars:
                    for k in s_vars[v]:
                        s_vars[v][k].fixed = o_flags[t, v, k]

        if blk.config.mixed_state_block is None:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(blk, tee=slc.tee)

            if not check_optimal_termination(res):
                raise InitializationError(
                    f"{blk.name} failed to initialize successfully. Please "
                    f"check the output logs for more information."
                )

            init_log.info(
                "Initialization Step 2 Complete: {}".format(idaeslog.condition(res))
            )
        else:
            init_log.info("Initialization Complete.")

        if hold_state is True:
            return flags
        else:
            blk.release_state(flags, outlvl=outlvl)
