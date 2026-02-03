#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
General purpose separator block for IDAES models
"""

from enum import Enum
from pandas import DataFrame

from pyomo.environ import (Block, Set)
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In, ListOf, Bool

from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
    MaterialBalanceType,
    MomentumBalanceType,
)
from idaes.core.util.config import (
    is_physical_parameter_block,
    is_state_block,
)
from idaes.core.util.exceptions import (
    BurntToast,
    ConfigurationError,
)
from idaes.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.units_of_measurement import report_quantity
from idaes.core.initialization import ModularInitializerBase

__author__ = "Team Ahuora"


# Set up logger
_log = idaeslog.getLogger(__name__)


# Enumerate options for balances
class SplittingType(Enum):
    """
    Enum of supported material split types.
    """

    totalFlow = 1
    phaseFlow = 2
    componentFlow = 3
    phaseComponentFlow = 4


class EnergySplittingType(Enum):
    """
    Enum of support energy split types.
    """

    none = 0
    equal_molar_enthalpy = 2


class SimpleSeparatorInitializer(ModularInitializerBase):
    """
    Initializer for Separator blocks.

    """

    def initialization_routine(
        self,
        model: Block,
    ):
        """
        Initialization routine for Separator Blocks.

        This routine starts by initializing the feed and outlet streams using simple rules.

        Args:
            model: model to be initialized

        Returns:
            None

        """
        init_log = idaeslog.getInitLogger(
            model.name, self.get_output_level(), tag="unit"
        )
        solve_log = idaeslog.getSolveLogger(
            model.name, self.get_output_level(), tag="unit"
        )

        # Create solver
        solver = self._get_solver()
        # Initialize mixed state block
        
        mblock = model.mixed_state
        self.get_submodel_initializer(mblock).initialize(mblock)

        res = None
        if degrees_of_freedom(model) != 0:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver.solve(model, tee=slc.tee)
                init_log.info(
                    "Initialization Step 1 Complete: {}".format(idaeslog.condition(res))
                )

        for c, s in component_status.items():
            if s:
                c.activate()


        # Initialize outlet StateBlocks
        outlet_list = model._create_outlet_list()

        # Initializing outlet states
        for o in outlet_list:
            # Get corresponding outlet StateBlock
            o_block = getattr(model, o + "_state")

            # Create dict to store fixed status of state variables
            for t in model.flowsheet().time:
                # Calculate values for state variables
                s_vars = o_block[t].define_state_vars()
                for var_name_port, var_obj in s_vars.items():
                    for k in var_obj:
                        # If fixed, use current value
                        # otherwise calculate guess from mixed state and fix
                        if not var_obj[k].fixed:
                            m_var = getattr(mblock[t], var_obj.local_name)
                            if "flow" in var_name_port:
                                # Leave initial value
                                pass
                            else:
                                # Otherwise intensive, equate to mixed stream
                                var_obj[k].set_value(m_var[k].value)

            # Call initialization routine for outlet StateBlock
            self.get_submodel_initializer(o_block).initialize(o_block)

        init_log.info("Initialization Complete.")

        return res


@declare_process_block_class("SimpleSeparator")
class SimpleSeparatorData(UnitModelBlockData):
    """
    This is a simple Splitter block with the IDAES modeling framework. 
    Unlike the generic Separator, this block avoids use of split fractions.

    This model creates a number of StateBlocks to represent the outgoing
    streams, then writes a set of phase-component material balances, an
    overall enthalpy balance (2 options), and a momentum balance (2 options)
    linked to a mixed-state StateBlock. The mixed-state StateBlock can either
    be specified by the user (allowing use as a sub-model), or created by the
    Separator.
    """

    default_initializer = SimpleSeparatorInitializer

    CONFIG = UnitModelBlockData.CONFIG()
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for mixer",
            doc="""Property parameter object used to define property
calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property
block(s) and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "outlet_list",
        ConfigValue(
            domain=ListOf(str),
            description="List of outlet names",
            doc="""A list containing names of outlets,
**default** - None.
**Valid values:** {
**None** - use num_outlets argument,
**list** - a list of names to use for outlets.}""",
        ),
    )
    CONFIG.declare(
        "num_outlets",
        ConfigValue(
            domain=int,
            description="Number of outlets to unit",
            doc="""Argument indicating number (int) of outlets to construct,
not used if outlet_list arg is provided,
**default** - None.
**Valid values:** {
**None** - use outlet_list arg instead, or default to 2 if neither argument
provided,
**int** - number of outlets to create (will be named with sequential integers
from 1 to num_outlets).}""",
        ),
    )
    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.useDefault.
**Valid values:** {
**MaterialBalanceType.useDefault - refer to property package for default
balance type
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
    **default** - MomentumBalanceType.pressureTotal.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - pressure in all outlets is equal,
    **MomentumBalanceType.pressurePhase** - not yet supported,
    **MomentumBalanceType.momentumTotal** - not yet supported,
    **MomentumBalanceType.momentumPhase** - not yet supported.}""",
        ),
    )
    CONFIG.declare(
        "has_phase_equilibrium",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Calculate phase equilibrium in mixed stream",
            doc="""Argument indicating whether phase equilibrium should be
calculated for the resulting mixed stream,
**default** - False.
**Valid values:** {
**True** - calculate phase equilibrium in mixed stream,
**False** - do not calculate equilibrium in mixed stream.}""",
        ),
    )

    def build(self):
        """
        General build method for SeparatorData. This method calls a number
        of sub-methods which automate the construction of expected attributes
        of unit models.

        Inheriting models should call `super().build`.

        Args:
            None

        Returns:
            None
        """
        # Call super.build()
        super(SimpleSeparatorData, self).build()

        # Call setup methods from ControlVolumeBlockData
        self._get_property_package()
        self._get_indexing_sets()

        # Create list of inlet names
        outlet_list = self._create_outlet_list()

        mixed_block = self._add_mixed_state_block()

        # Add inlet port
        self._add_inlet_port_objects(mixed_block)

        # Build StateBlocks for outlet
        outlet_blocks = self._add_outlet_state_blocks(outlet_list)
        self.outlet_idx = Set(initialize=outlet_list)

        # Construct splitting equations
        self._add_material_balance(mixed_block, outlet_blocks)
        self._add_energy_balance(mixed_block, outlet_blocks)
        self._add_momentum_balance(mixed_block, outlet_blocks)

        # Construct outlet port objects
        self._add_outlet_port_objects(outlet_list)

    def _create_outlet_list(self):
        """
        Create list of outlet stream names based on config arguments.

        Returns:
            list of strings
        """
        if self.config.outlet_list is not None and self.config.num_outlets is not None:
            # If both arguments provided and not consistent, raise Exception
            if len(self.config.outlet_list) != self.config.num_outlets:
                raise ConfigurationError(
                    "{} Separator provided with both outlet_list and "
                    "num_outlets arguments, which were not consistent ("
                    "length of outlet_list was not equal to num_outlets). "
                    "Please check your arguments for consistency, and "
                    "note that it is only necessry to provide one of "
                    "these arguments.".format(self.name)
                )
        elif self.config.outlet_list is None and self.config.num_outlets is None:
            # If no arguments provided for outlets, default to num_outlets = 2
            self.config.num_outlets = 2

        # Create a list of names for outlet StateBlocks
        if self.config.outlet_list is not None:
            outlet_list = self.config.outlet_list
        else:
            outlet_list = [
                "outlet_" + str(n) for n in range(1, self.config.num_outlets + 1)
            ]

        return outlet_list

    def _add_outlet_state_blocks(self, outlet_list):
        """
        Construct StateBlocks for all outlet streams.

        Args:
            list of strings to use as StateBlock names

        Returns:
            list of StateBlocks
        """
        # Setup StateBlock argument dict
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["defined_state"] = False

        # Create empty list to hold StateBlocks for return
        outlet_blocks = []

        # Create an instance of StateBlock for all outlets
        for o in outlet_list:
            o_obj = self.config.property_package.build_state_block(
                self.flowsheet().time, doc="Material properties at outlet", **tmp_dict
            )

            setattr(self, o + "_state", o_obj)

            outlet_blocks.append(getattr(self, o + "_state"))

        return outlet_blocks

    def _add_mixed_state_block(self):
        """
        Constructs StateBlock to represent mixed stream.

        Returns:
            New StateBlock object
        """
        # Setup StateBlock argument dict
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["defined_state"] = True

        self.mixed_state = self.config.property_package.build_state_block(
            self.flowsheet().time, doc="Material properties of mixed stream", **tmp_dict
        )

        return self.mixed_state

    def _add_inlet_port_objects(self, mixed_block):
        """ Adds inlet Port object."""
        self.add_port(name="inlet", block=mixed_block, doc="Inlet Port")

    def _add_outlet_port_objects(self, outlet_list):
        """Adds outlet Port objects."""
        for p in outlet_list:
            o_state = getattr(self, p + "_state")
            self.add_port(name=p, block=o_state, doc="Outlet Port")

    def _add_material_balance(self, mixed_block, outlet_blocks):
        """Add overall material balance equation."""
        # Get phase component list(s)
        pc_set = mixed_block.phase_component_set
        
        # Write phase-component balances
        @self.Constraint(self.flowsheet().time, doc="Material balance equation")
        def material_balance_equation(b, t):
            return 0 == sum(
                sum(
                    mixed_block[t].get_material_flow_terms(p, j)
                    - 
                    sum(
                        o[t].get_material_flow_terms(p, j)
                        for o in outlet_blocks
                    )                    
                    for j in mixed_block.component_list
                    if (p, j) in pc_set
                )
                for p in mixed_block.phase_list
            )

    def _add_energy_balance(self, mixed_block, outlet_blocks):
        """
        Creates constraints for splitting the energy flows.
        """
        # split basis is equal_molar_enthalpy
        @self.Constraint(
            self.flowsheet().time,
            self.outlet_idx,
            doc="Molar enthalpy equality constraint",
        )
        def molar_enthalpy_equality_eqn(b, t, o):
            o_block = getattr(self, o + "_state")
            return mixed_block[t].enth_mol == o_block[t].enth_mol

    def _add_momentum_balance(self, mixed_block, outlet_blocks):
        """
        Creates constraints for splitting the momentum flows - done by equating
        pressures in outlets.
        """
        if self.config.momentum_balance_type is MomentumBalanceType.pressureTotal:
            @self.Constraint(
                self.flowsheet().time,
                self.outlet_idx,
                doc="Pressure equality constraint",
            )
            def pressure_equality_eqn(b, t, o):
                o_block = getattr(self, o + "_state")
                return mixed_block[t].pressure == o_block[t].pressure

    def model_check(blk):
        """
        This method executes the model_check methods on the associated state
        blocks (if they exist). This method is generally called by a unit model
        as part of the unit's model_check method.

        Args:
            None

        Returns:
            None
        """
        # Try property block model check
        for t in blk.flowsheet().time:
            try:
                blk.mixed_state[t].model_check()
            except AttributeError:
                _log.warning(
                    "{} Separator inlet state block has no "
                    "model check. To correct this, add a "
                    "model_check method to the associated "
                    "StateBlock class.".format(blk.name)
                )

            try:
                outlet_list = blk._create_outlet_list()
                for o in outlet_list:
                    o_block = getattr(blk, o + "_state")
                    o_block[t].model_check()
            except AttributeError:
                _log.warning(
                    "{} Separator outlet state block has no "
                    "model checks. To correct this, add a model_check"
                    " method to the associated StateBlock class.".format(blk.name)
                )

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

        mblock = blk.mixed_state
        flags = mblock.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            hold_state=True,
        )

        if degrees_of_freedom(blk) != 0:
            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(blk, tee=slc.tee)
                init_log.info(
                    "Initialization Step 1 Complete: {}".format(idaeslog.condition(res))
                )

        # Initialize outlet StateBlocks
        outlet_list = blk._create_outlet_list()

        # Premises for initializing outlet states:
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
                                # Leave initial value, but avoid negative flows
                                if s_vars[v][k].value < 1e-4:
                                    s_vars[v][k].set_value(1e-2)
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

        init_log.info("Initialization Complete.")
        return flags

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        """
        Method to release state variables fixed during initialization.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state = True.
            outlvl : sets output level of logging

        Returns:
            None
        """
        mblock = blk.mixed_state
        mblock.release_state(flags, outlvl=outlvl)

    def calculate_scaling_factors(self):
        mb_type = self.config.material_balance_type
        mixed_state = self.mixed_state
        if mb_type == MaterialBalanceType.useDefault:
            t_ref = self.flowsheet().time.first()
            mb_type = mixed_state[t_ref].default_material_balance_type()
        super().calculate_scaling_factors()

        if hasattr(self, "temperature_equality_eqn"):
            for (t, i), c in self.temperature_equality_eqn.items():
                s = iscale.get_scaling_factor(
                    mixed_state[t].temperature, default=1, warning=True
                )
                iscale.constraint_scaling_transform(c, s)

        if hasattr(self, "pressure_equality_eqn"):
            for (t, i), c in self.pressure_equality_eqn.items():
                s = iscale.get_scaling_factor(
                    mixed_state[t].pressure, default=1, warning=True
                )
                iscale.constraint_scaling_transform(c, s)

        if hasattr(self, "material_splitting_eqn"):
            if mb_type == MaterialBalanceType.componentPhase:
                for (t, _, p, j), c in self.material_splitting_eqn.items():
                    flow_term = mixed_state[t].get_material_flow_terms(p, j)
                    s = iscale.get_scaling_factor(flow_term, default=1)
                    iscale.constraint_scaling_transform(c, s)
            elif mb_type == MaterialBalanceType.componentTotal:
                for (t, _, j), c in self.material_splitting_eqn.items():
                    s = None
                    for p in mixed_state.phase_list:
                        try:
                            ft = mixed_state[t].get_material_flow_terms(p, j)
                        except KeyError:
                            # This component does not exist in this phase
                            continue
                        if s is None:
                            s = iscale.get_scaling_factor(ft, default=1)
                        else:
                            _s = iscale.get_scaling_factor(ft, default=1)
                            s = _s if _s < s else s
                    iscale.constraint_scaling_transform(c, s)
            elif mb_type == MaterialBalanceType.total:
                pc_set = mixed_state.phase_component_set
                for (t, _), c in self.material_splitting_eqn.items():
                    for i, (p, j) in enumerate(pc_set):
                        ft = mixed_state[t].get_material_flow_terms(p, j)
                        if i == 0:
                            s = iscale.get_scaling_factor(ft, default=1)
                        else:
                            _s = iscale.get_scaling_factor(ft, default=1)
                            s = _s if _s < s else s
                    iscale.constraint_scaling_transform(c, s)

    def _get_performance_contents(self, time_point=0):
        if hasattr(self, "split_fraction"):
            var_dict = {}
            for k, v in self.split_fraction.items():
                if k[0] == time_point:
                    var_dict[f"Split Fraction [{str(k[1:])}]"] = v
            return {"vars": var_dict}
        else:
            return None

    def _get_stream_table_contents(self, time_point=0):
        outlet_list = self._create_outlet_list()

        io_dict = {}
        io_dict["Inlet"] = self.mixed_state

        for o in outlet_list:
            io_dict[o] = getattr(self, o + "_state")

        return create_stream_table_dataframe(io_dict, time_point=time_point)
