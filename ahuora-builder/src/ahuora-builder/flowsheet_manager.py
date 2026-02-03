from collections import defaultdict, deque
import re
from pyomo.network import SequentialDecomposition, Port
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    SolverFactory,
    Block,
    Expression,
    Constraint,
    Objective,
    minimize,
    assert_optimal_termination,
    ScalarVar,
    units as pyunits,
    TerminationCondition
)
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import report_statistics, degrees_of_freedom
import idaes.logger as idaeslog
from idaes.core.util import DiagnosticsToolbox
from ahuora_builder_types import FlowsheetSchema
from ahuora_builder_types.flowsheet_schema import SolvedFlowsheetSchema
from .property_package_manager import PropertyPackageManager
from .port_manager import PortManager
from .arc_manager import ArcManager
from .tear_manager import TearManager
from .unit_model_manager import UnitModelManager
from .methods.adapter_library import AdapterLibrary
from .methods.adapter import (
    serialize_properties_map,
    deactivate_fixed_guesses,
    deactivate_component,
    deactivate_components,
    add_corresponding_constraint,
)
from .methods.expression_parsing import parse_expression, ExpressionParsingError
from .timing import start_timing
from .methods.units_handler import get_value, get_attached_unit, check_units_equivalent
from .properties_manager import PropertiesManager
from .custom.energy.power_property_package import PowerParameterBlock
from .custom.energy.ac_property_package import acParameterBlock
from .custom.energy.transformer_property_package import transformerParameterBlock 
from pyomo.core.base.units_container import units, _PyomoUnit
from idaes.core.util.model_serializer import StoreSpec, from_json, to_json
from .diagnostics.infeasibilities import print_infeasibilities

# CSTR Imports from example flowsheet. 
# To be used as placeholders until bespoke functions can be developed.
from .custom import hda_reaction as reaction_props
from .custom.hda_ideal_VLE import HDAParameterBlock
from ..packages.properties_manager import PropertyComponent
from property_packages.build_package import build_package

# from amplpy import modules
# Import required to allow the library to set the PATH and allow conopt to be found.

def build_flowsheet(dynamic=False,time_set=[0]):
    """
    Builds a flowsheet block
    """
    # create the model and the flowsheet block
    model = ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=dynamic, time_set=time_set, time_units=units.s)
    model.fs.guess_vars = []
    model.fs.controlled_vars = Block()
    # Always add the power property package (as there's only one and there's no different property package types)
    # TODO: should this be on-demand?
    model.fs.power_pp = PowerParameterBlock()
    model.fs.ac_pp = acParameterBlock()
    model.fs.tr_pp = transformerParameterBlock()
    # properties map: { id: pyomo variable or expression }
    # used to map symbols for a sympy expression
    model.fs.properties_map = PropertiesManager()
    # list of component constraints to add later
    model.fs.constraint_exprs = []

    # Placholder property packages for CSTR reactor.
    model.fs.BTHM_params = HDAParameterBlock()
    #Hard-coded peng-robinson package for reactor:
    model.fs.peng_robinson = build_package("peng-robinson",["benzene", "toluene", "hydrogen", "methane"], ["Liq","Vap"])
    
    # Reaction package for the HDA reaction
    model.fs.reaction_params = reaction_props.HDAReactionParameterBlock(
       property_package=model.fs.peng_robinson
    )

    return model


class FlowsheetManager:
    """
    Manages the flowsheet, including the property packages, unit models, ports, arcs, and tears
    Includes methods to load, initialise, and solve the flowsheet
    """

    def __init__(self, schema: FlowsheetSchema) -> None:
        """
        Stores all relevant information about the flowsheet, without actually loading it
        """
        self.timing = start_timing()
        self.timing.add_timing("initialise_flowsheet_manager")

        self.model = build_flowsheet(dynamic=schema.dynamic, time_set=schema.time_set)

        self.schema = schema
        # Add property packages first, so that unit models can use them
        self.property_packages = PropertyPackageManager(self)
        # Add the port manager, so the unit models can register their ports
        self.ports = PortManager()
        # Add unit models
        self.unit_models = UnitModelManager(self)
        # Add arcs to connect the unit models together
        self.arcs = ArcManager(self)
        # set certain arcs as tears
        self.tears = TearManager(self)

    def load(self) -> None:
        """
        Parses the schema and loads the model
        """
        self.timing.step_into("load_flowsheet")
        # Load property packages first, so that unit models can use them
        self.timing.add_timing("load_property_packages")
        self.property_packages.load()
        self.timing.step_into("load_unit_models")
        self.unit_models.load()
        self.timing.step_out()
        # no need to load ports seperately, they are loaded by the unit models
        # Load arcs to connect the unit models together
        self.arcs.load()
        # load any expressions
        self.load_specs()
        # if dynamics, apply finite difference transformaiton
        if self.schema.dynamic:
            print("performing finite difference with", len(self.model.fs.time), "time steps")
            TransformationFactory("dae.finite_difference").apply_to(
                self.model.fs, 
                nfe=len(self.model.fs.time)-1, # Number of finite elements to use for discretization. We aren't adding any extra steps as our constraints dont work for that.
                wrt=self.model.fs.time, 
                scheme="BACKWARD"
            )
            

        self.timing.step_out()

    def load_specs(self) -> None:
        """
        Loads expressions from the schema
        """
        fs = self.model.fs
        specs = self.schema.expressions or []

        ## Build dependency tree and sort specs before loading.
        dependencies, result_ids = self.build_spec_dependency_tree(specs)
        sorted_result_ids = self.topological_sort(dependencies, result_ids)

        # load the specs (expressions within specifications tab)
        for result_id in sorted_result_ids:
            spec_config = next(spec for spec in specs if spec["id"] == result_id)
            expression_str = spec_config["expression"]
            try:
                component_name = f"{spec_config['name']}_{spec_config['id']}"
                def expression_rule(blk, time_index):
                    return parse_expression(expression_str, fs,time_index)
                
                component = Expression(fs.time, rule=expression_rule)
                fs.add_component(component_name, component)
                fs.properties_map.add(
                    spec_config["id"], component, component.name, unknown_units=True
                )
            except ExpressionParsingError as e:
                raise ExpressionParsingError(f"{e} when parsing expression '{expression_str}' for {component_name}: ")

        # load constraints (expressions for specific property infos)
        # can only handle equality constraints for now 
        for component, expr_str, id in fs.constraint_exprs:
            # get the time index
            #for time_index in fs.time:
            if component.index_set().dimen != 0 and component.index_set() != fs.time:
                raise ExpressionParsingError(f"Cannot add constraint for {component}: only time-indexed components are supported.") 
            try:
                def constraint_rule(blk, time_index):
                    expression = parse_expression(expr_str, fs,time_index)
                    # make sure the units of the expression are the same as the component
                    u1, u2 = get_attached_unit(component), get_attached_unit(expression)
                    if not check_units_equivalent(u1, u2):
                        raise ValueError(
                            f"Failed to add constraint for {component}: units do not match (expected {u1}, got {u2})"
                        )
                    return pyunits.convert(component[time_index], to_units=u2) == expression
                c = Constraint(component.index_set(), rule= constraint_rule)
                name = f"equality_constraint_{id}"
                fs.add_component(name, c)
                add_corresponding_constraint(fs, c, id)
            except ExpressionParsingError as e:
                raise ExpressionParsingError(f"Failed to parse constraint expression '{expr_str}'  for {component}: {expr_str}, error: {e}")


    def build_spec_dependency_tree(self, specs) -> tuple:
        """
        Builds dependency tree for expressions based on the references in their respective expressions.
        """
        dependencies = defaultdict(
            set
        )  # Maps an expression's result_id to a set of result_ids it depends on
        result_ids = set()  # A set to track all result_ids

        # Get a list of all result_id's.
        for spec in specs:
            result_id = spec["id"]
            result_ids.add(result_id)

        for spec in specs:
            result_id = spec["id"]
            expression = spec["expression"]

            # Find the result_ids that this expression depends on
            dependent_expressions = self.get_dependent_expressions(
                expression, result_ids
            )

            if dependent_expressions:
                # If the expression depends on another result_id, add dependency
                for id in dependent_expressions:
                    dependencies[id].add(result_id)

        return dependencies, result_ids

    def get_dependent_expressions(self, expression: str, all_result_ids: set) -> list:
        """
        Gets all result_ids referenced in the expression.
        """
        # match result_ids starting with 'id_' followed by numbers
        ids = re.findall(r"\b(id_\d+)\b", expression)

        # Filter the referenced_ids to only include those that are in all_result_ids
        valid_referenced_ids = []
        for id in ids:
            # get the numeric part after "id_" and check if it's in the all_result_ids
            numeric_id = int(id[3:])
            if numeric_id in all_result_ids:
                valid_referenced_ids.append(numeric_id)

        return valid_referenced_ids

    def topological_sort(self, dependencies: dict, result_ids: set) -> list:
        """
        Performs topological sorting on the specification dependency tree.
        """
        # Track teh in-degree count for all expressions (edges coming into it)
        in_degree = defaultdict(int)

        # Count dependencies for each expression
        for result_id in result_ids:
            for dep in dependencies[result_id]:
                in_degree[dep] += 1

        # Initialise the queue with result_ids that have no dependencies (in-degree 0)
        dequeue = deque(
            [result_id for result_id in result_ids if in_degree[result_id] == 0]
        )

        sorted_result_ids = []

        while dequeue:
            result_id = dequeue.popleft()
            sorted_result_ids.append(result_id)

            # loop thtough and decrement each dependent expression's in_degree, and append it to the deque if it has an in-degree of 0.
            for dependent_result_id in dependencies[result_id]:
                in_degree[dependent_result_id] -= 1
                if in_degree[dependent_result_id] == 0:
                    dequeue.append(dependent_result_id)

        # If there are any result_ids left with non-zero in-degree, a cycle exists so error.
        if len(sorted_result_ids) != len(result_ids):
            raise ValueError(
                "Cycle detected in the dependency graph. Check an expression does not reference itself!"
            )

        return sorted_result_ids

    def initialise(self) -> None:
        """
        Expands the arcs and initialises the model
        """
        # check if initialisation is disabled for this scenario
        if getattr(self.schema, "disable_initialization", False):
            # if disable initialisation is set to True, then we don't need to initialise the model
            print("Initialisation is disabled for this scenario.")

        # We need to "expand_arcs" to make them a bidirection link that actually imposes constraints on the model.
        TransformationFactory("network.expand_arcs").apply_to(self.model)

        self.timing.step_into("initialise_model")

        # load tear guesses (including var/constraint unfixing & deactivation and/or equality constraint deactivation)
        self.tears.load()

        tears = self.tears._tears

        def init_unit(unit):
            if getattr(self.schema, "disable_initialization", False):
                return

            print(f"Initializing unit {unit}")
            self.timing.add_timing(f"init_{unit.name}")
            #unit.display()
            unit.initialize(outlvl=idaeslog.INFO)
            #unit.report()

        self.timing.add_timing("setup_sequential_decomposition")
        # Use SequentialDecomposition to initialise the model
        seq = SequentialDecomposition(
            #run_first_pass=True,
            iterLim=1,
        )
        seq.set_tear_set(tears)
        # use create_graph to get the order of sequential decomposition, and also to
        # find any units that are not connected to the sequential decomposition
        G = seq.create_graph(self.model)
        order = seq.calculation_order(G)
        seq_blocks = []
        for o in order:
            seq_blocks.append(o[0])
        print("Order of initialisation:", [blk.name for blk in seq_blocks])
        # set all the tear guesses before running the decomposition
        for arc in tears:
            port = arc.destination
            # guesses used are initial values for each var
            guesses = {}
            guesses = {key: get_value(var) for key, var in port.vars.items()}
            print(f"Guess for {port}: {guesses}")

        self.timing.step_into("run_sequential_decomposition")

        # sequential decomposition completes when all vars across port
        # equalities are within tol of each other
        seq.options["tol"] = 1e-2
        # seq.options["solve_tears"] = False
        res = seq.run(self.model, init_unit)

        self.timing.step_out()
        self.timing.add_timing("initialise_disconnected_units")
        # Initialise any unit model that is not connected to the sequential decomposition
        for blk in self.model.fs.component_data_objects(
            Block, descend_into=False, active=True
        ):
            ports = list(blk.component_objects(Port, descend_into=False))
            if len(ports) == 0:
                continue  # if the block has no ports, then it is not a unit model
            if blk in seq_blocks:
                continue  # already initialised by sequential decomposition
            init_unit(blk)

        # unfix guess vars
        deactivate_fixed_guesses(self.model.fs.guess_vars)

        self.timing.step_out()

    def serialise(self) -> SolvedFlowsheetSchema:
        self.timing.add_timing("serialise_model")

        initial_values = {}
        for unit_model_id, unit_model in self.unit_models._unit_models.items():
            initial_values[str(unit_model_id)] = to_json(unit_model, return_dict=True, wts=StoreSpec.value())

        solved_flowsheet = SolvedFlowsheetSchema(
            id=self.schema.id,
            properties=serialize_properties_map(self.model.fs),
            initial_values=initial_values
        )

        return solved_flowsheet

    def report_statistics(self) -> None:
        """
        Reports statistics about the model
        """
        report_statistics(self.model)
        # I think the diagnostics toolbox is taking too long to run, so commenting it out for now.
        # dt = DiagnosticsToolbox(self.model)
        # dt.report_structural_issues()
        # dt.display_overconstrained_set()
        # dt.display_underconstrained_set()
    
    def diagnose_problems(self) -> None:
        print("=== DIAGNOSTICS ===")
        report_statistics(self.model)
        dt = DiagnosticsToolbox(self.model)
        dt.report_structural_issues()
        dt.display_overconstrained_set()
        dt.display_underconstrained_set()
        #dt.display_components_with_inconsistent_units()
        try:
            dt.compute_infeasibility_explanation()
        except Exception as e:
            print(f"{e}") # error is probably because it is feasible
        dt.report_numerical_issues()
        dt.display_near_parallel_constraints()
        dt.display_variables_at_or_outside_bounds()
        print("=== END DIAGNOSTICS ===")

    def degrees_of_freedom(self) -> int:
        """
        Returns the degrees of freedom of the model
        """
        return int(degrees_of_freedom(self.model))

    def check_model_valid(self) -> None:
        """
        Checks if the model is valid by checking the
        degrees of freedom. Will raise an exception if
        the model is not valid.
        """
        self.timing.add_timing("check_model_valid")
        degrees_of_freedom = self.degrees_of_freedom()
        if degrees_of_freedom != 0:
            #self.model.display()  # prints the vars/constraints for debugging
            raise Exception(
                f"Degrees of freedom is not 0. Degrees of freedom: {degrees_of_freedom}"
            )

    def solve(self) -> None:
        """
        Solves the model
        """
        self.timing.add_timing("solve_model")
        print("=== Starting Solve ===")

        opt = SolverFactory(self.schema.solver_option)
        # opt.options["max_iter"] = 5000

        if self.schema.solver_option != "conopt":
            opt.options["max_iter"] = 1000
        try:
            res = opt.solve(self.model, tee=True)
            if res.solver.termination_condition != TerminationCondition.optimal:
                print_infeasibilities(self.model.fs.properties_map)
            assert_optimal_termination(res)
        except ValueError as e:
            if str(e).startswith("No variables appear"):
                # https://github.com/Pyomo/pyomo/pull/3445
                pass
            else:
                raise e

    def optimize(self) -> None:
        if self.schema.optimizations is None or self.schema.optimizations == []:
            return

        self.timing.add_timing("optimize_model")
        print("=== Starting Optimization ===")

        # ipopt doesn't support multiple objectives, so we need to create
        # a single objective expression.
        # this is done by summing all objectives in the model, adding
        # or subtracting based on the sense (minimize or maximize)
        objective_expr = 0
        for schema in self.schema.optimizations:
            # get the expression component to optimize
            # TODO: This is assuming optimisation is run on a steady-state simulation. We need to change how this works to handle dynamics.
            # For now, just hardcoding time_index=0
            objective_component = self.model.fs.properties_map.get_component(schema.objective)
            # the objective component should be a scalar in non-dynamic models
            # in a dynamic model it'll be indexed across all time steps
            # We sum up all time steps so each time step is weighted equally.
            objective = sum(objective_component.values())
            # add or subtract the objective based on the sense
            sense = schema.sense
            if sense == "minimize":
                objective_expr += objective
            else:
                objective_expr -= objective



            # unfix relevant vars (add to degrees of freedom)
            for dof_info in schema.unfixed_variables:
                id = dof_info.id # Id of the propertyValue for this degree of freedom
                var: PropertyComponent = self.model.fs.properties_map.get(id)
        

                # TODO: may need to handle deactivating constraints,
                # for expressions that are constrained (instead of state vars)
                c = var.corresponding_constraint
                if c is not None:
                    # TODO: better typing for constraints
                    if isinstance(c, ScalarConstraint) or isinstance(c, IndexedConstraint):
                        c.deactivate()
                    else:
                        deactivate_components(c)
                else:
                    for i in var.component.values():
                        if isinstance(i, ScalarVar):
                            i.unfix()
                        # Because if not, it is ExpressionData, meaning it is already an expression and doesn't need to be unfixed. (we've already checked if there is a constraint for it above too.)

                
                # TODO: set attributes for upper and lower bounds of property infos. i.e. use propertyinfo id.
                # Var is either a Variable or Expression
                # set the minimum or maximum bounds for this variable if they are enabled
                #self.model.upper_bound_12 =   Constraint(expr= var <= upper_bound_value )

                upper_bound = dof_info.upper_bound
                lower_bound = dof_info.lower_bound

                c = var.component

                if upper_bound is not None:
                    def upper_bound_rule(model,index):
                        return c[index] <= upper_bound
                    upper_bound_constraint = Constraint(c.index_set(),rule=upper_bound_rule)
                    setattr(self.model,"upper_bound_" + str(id), upper_bound_constraint)

                if lower_bound is not None:
                    def lower_bound_rule(model,index):
                        return c[index] >= lower_bound
                    lower_bound_constraint = Constraint(c.index_set(),rule=lower_bound_rule)
                    setattr(self.model,"lower_bound_" + str(id), lower_bound_constraint)


        # add the objective to the model
        self.model.objective = Objective(expr=objective_expr, sense=minimize)

        # solve the model with the objective
        opt = SolverFactory(self.schema.solver_option)

        if self.schema.solver_option != "conopt":
            opt.options["max_iter"] = 1000

        try:
            res = opt.solve(self.model, tee=True)
            assert_optimal_termination(res)
        except ValueError as e:
            if str(e).startswith("No variables appear"):
                # https://github.com/Pyomo/pyomo/pull/3445
                pass
            else:
                raise e
