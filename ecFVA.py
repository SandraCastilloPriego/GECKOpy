from cobra import Gene, Model, Reaction
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union
import cobra
import pandas as pd
import numpy as np
from optlang.symbolics import Zero

import multiprocessing
import os
import pickle
from pathlib import Path
from platform import system
from tempfile import mkstemp
from types import TracebackType
from typing import Any, Callable, Type
import logging
import re
from functools import partial
from types import ModuleType
from typing import Dict, List, NamedTuple
from warnings import warn

import optlang
import pandas as pd
from optlang.interface import (
    FEASIBLE,
    INFEASIBLE,
    ITERATION_LIMIT,
    NUMERIC,
    OPTIMAL,
    SUBOPTIMAL,
    TIME_LIMIT,
)
from optlang.symbolics import Basic, Zero

from cobra.exceptions import (
    OPTLANG_TO_EXCEPTIONS_DICT,
    OptimizationError,
    SolverNotFound,
)
from cobra.util.context import get_context

"""Provide variability based methods such as flux variability or gene essentiality."""

from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from cobra import Gene, Model, Reaction


def map_fluxes(fva):   
    names=[k[0:6] for k in fva.index]    
    unique_ids=list(set(names))
    fluxes={}
    for n in unique_ids:
        rows= fva[fva.index.str.contains(n)]
        rev=0
        ford=0
        if len(rows)>1:
            for key, row in rows.iterrows():
                if "_REV" in key:
                    rev= rev+row["maximum"]
                    ford = ford+ row["minimum"]
                else:
                    rev = rev+ row["minimum"]
                    ford = ford+ row["maximum"]
            fluxes[n]=[rev*-1, ford]  
        else:
            fluxes[n]=[rows.iloc[0]["minimum"], rows.iloc[0]["maximum"]]  
    newfva=pd.DataFrame(fluxes).transpose().sort_index()
    newfva.columns=["minimum","maximum"]
    return newfva
        
    
def get_similar_reactions(idr,model):
    similar=[]
    for rxn in model.reactions:
        if idr in rxn.id:
            similar.append(rxn)

    return similar 
def _init_worker(model: "Model", loopless: bool, sense: str) -> None:
    """Initialize a global model object for multiprocessing.

    Parameters
    ----------
    model: cobra.Model
        The model to operate on.
    loopless: bool
        Whether to use loopless version.
    sense: {"max", "min"}
        Whether to maximise or minimise objective.

    """
    global _model
    global _loopless
    _model = model
    _model.solver.objective.direction = sense
    _loopless = loopless


def _fva_step(reaction_id: str) -> Tuple[str, float]:
    """Take a step for calculating FVA.

    Parameters
    ----------
    reaction_id: str
        The ID of the reaction.

    Returns
    -------
    tuple of (str, float)
        The reaction ID with the flux value.

    """
    global _model
    global _loopless
    rxn = _model.reactions.get_by_id(reaction_id)
    # The previous objective assignment already triggers a reset
    # so directly update coefs here to not trigger redundant resets
    # in the history manager which can take longer than the actual
    # FVA for small models   
    if "_EXP" in rxn.id:
        similar=get_similar_reactions(rxn.id[0:6], _model)
        if "_REV" in rxn.id:
            for sim in similar:
                if "_REV" in rxn.id:
                    _model.solver.objective.set_linear_coefficients(
                    {rxn.forward_variable: 1, rxn.reverse_variable: -1}
                    )                
        else:
            for sim in similar:
                if "_REV" not in rxn.id:
                    _model.solver.objective.set_linear_coefficients(
                    {rxn.forward_variable: 1, rxn.reverse_variable: -1}
                    )
    else:       
        _model.solver.objective.set_linear_coefficients(
            {rxn.forward_variable: 1, rxn.reverse_variable: -1}
        )
    _model.slim_optimize()
    check_solver_status(_model.solver.status)
    if _loopless:
        value = loopless_fva_iter(_model, rxn)
    else:
        value = _model.solver.objective.value
    # handle infeasible case
    if value is None:
        value = float("nan")
        logger.warning(
            f"Could not get flux for reaction {rxn.id}, setting it to NaN. "
            "This is usually due to numerical instability."
        )
    _model.solver.objective.set_linear_coefficients(
        {rxn.forward_variable: 0, rxn.reverse_variable: 0}
    )    
    return reaction_id, value


def flux_variability_analysis(
    model: "Model",
    reaction_list: Optional[List[Union["Reaction", str]]] = None,
    loopless: bool = False,
    fraction_of_optimum: float = 1.0,
    pfba_factor: Optional[float] = None,
    processes: Optional[int] = None,
    combine: bool = True
) -> pd.DataFrame:
    """Determine the minimum and maximum flux value for each reaction.

    Parameters
    ----------
    model : cobra.Model
        The model for which to run the analysis. It will *not* be modified.
    reaction_list : list of cobra.Reaction or str, optional
        The reactions for which to obtain min/max fluxes. If None will use
        all reactions in the model (default None).
    loopless : bool, optional
        Whether to return only loopless solutions. This is significantly
        slower. Please also refer to the notes (default False).
    fraction_of_optimum : float, optional
        Must be <= 1.0. Requires that the objective value is at least the
        fraction times maximum objective value. A value of 0.85 for instance
        means that the objective has to be at least at 85% percent of its
        maximum (default 1.0).
    pfba_factor : float, optional
        Add an additional constraint to the model that requires the total sum
        of absolute fluxes must not be larger than this value times the
        smallest possible sum of absolute fluxes, i.e., by setting the value
        to 1.1 the total sum of absolute fluxes must not be more than
        10% larger than the pFBA solution. Since the pFBA solution is the
        one that optimally minimizes the total flux sum, the `pfba_factor`
        should, if set, be larger than one. Setting this value may lead to
        more realistic predictions of the effective flux bounds
        (default None).
    processes : int, optional
        The number of parallel processes to run. If not explicitly passed,
        will be set from the global configuration singleton (default None).

    Returns
    -------
    pandas.DataFrame
        A data frame with reaction identifiers as the index and two columns:
        - maximum: indicating the highest possible flux
        - minimum: indicating the lowest possible flux

    Notes
    -----
    This implements the fast version as described in [1]_. Please note that
    the flux distribution containing all minimal/maximal fluxes does not have
    to be a feasible solution for the model. Fluxes are minimized/maximized
    individually and a single minimal flux might require all others to be
    sub-optimal.

    Using the loopless option will lead to a significant increase in
    computation time (about a factor of 100 for large models). However, the
    algorithm used here (see [2]_) is still more than 1000x faster than the
    "naive" version using `add_loopless(model)`. Also note that if you have
    included constraints that force a loop (for instance by setting all fluxes
    in a loop to be non-zero) this loop will be included in the solution.

    References
    ----------
    .. [1] Computationally efficient flux variability analysis.
       Gudmundsson S, Thiele I.
       BMC Bioinformatics. 2010 Sep 29;11:489.
       doi: 10.1186/1471-2105-11-489, PMID: 20920235

    .. [2] CycleFreeFlux: efficient removal of thermodynamically infeasible
       loops from flux distributions.
       Desouki AA, Jarre F, Gelius-Dietrich G, Lercher MJ.
       Bioinformatics. 2015 Jul 1;31(13):2159-65.
       doi: 10.1093/bioinformatics/btv096.

    """
    if reaction_list is None:
        reaction_ids = [r.id for r in model.reactions]
    else:
        reaction_ids = [r.id for r in model.reactions.get_by_any(reaction_list)]

    if processes is None:
        processes = 4
    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    fva_result = pd.DataFrame(
        {
            "minimum": np.zeros(num_reactions, dtype=float),
            "maximum": np.zeros(num_reactions, dtype=float),
        },
        index=reaction_ids,
    )
    prob = model.problem
    with model:
        # Safety check before setting up FVA.
        model.slim_optimize(
            error_value=None,
            message="There is no optimal solution for the chosen objective!",
        )
        # Add the previous objective as a variable to the model then set it to
        # zero. This also uses the fraction to create the lower/upper bound for
        # the old objective.
        # TODO: Use utility function here (fix_objective_as_constraint)?
        if model.solver.objective.direction == "max":
            fva_old_objective = prob.Variable(
                "fva_old_objective",
                lb=fraction_of_optimum * model.solver.objective.value,
            )
        else:
            fva_old_objective = prob.Variable(
                "fva_old_objective",
                ub=fraction_of_optimum * model.solver.objective.value,
            )
        fva_old_obj_constraint = prob.Constraint(
            model.solver.objective.expression - fva_old_objective,
            lb=0,
            ub=0,
            name="fva_old_objective_constraint",
        )
        model.add_cons_vars([fva_old_objective, fva_old_obj_constraint])

        if pfba_factor is not None:
            if pfba_factor < 1.0:
                warn(
                    "The 'pfba_factor' should be larger or equal to 1.",
                    UserWarning,
                )
            with model:
                add_pfba(model, fraction_of_optimum=0)
                ub = model.slim_optimize(error_value=None)
                flux_sum = prob.Variable("flux_sum", ub=pfba_factor * ub)
                flux_sum_constraint = prob.Constraint(
                    model.solver.objective.expression - flux_sum,
                    lb=0,
                    ub=0,
                    name="flux_sum_constraint",
                )
            model.add_cons_vars([flux_sum, flux_sum_constraint])

        model.objective = Zero  # This will trigger the reset as well
        for what in ("minimum", "maximum"):
            if processes > 1:
                # We create and destroy a new pool here in order to set the
                # objective direction for all reactions. This creates a
                # slight overhead but seems the most clean.
                chunk_size = len(reaction_ids) // processes
                with ProcessPool(
                    processes,
                    initializer=_init_worker,
                    initargs=(model, loopless, what[:3]),
                ) as pool:
                    for rxn_id, value in pool.imap_unordered(
                        _fva_step, reaction_ids, chunksize=chunk_size
                    ):
                        fva_result.at[rxn_id, what] = value
            else:
                _init_worker(model, loopless, what[:3])
                for rxn_id, value in map(_fva_step, reaction_ids):
                    fva_result.at[rxn_id, what] = value

    fva=fva_result[["minimum", "maximum"]]

    if combine:
        return map_fluxes(fva)
    else:
        return fva
    
   
 
 
 
"""Provide a process pool with enhanced performance on Windows."""




__all__ = ("ProcessPool",)


def _init_win_worker(filename: str) -> None:
    """Retrieve worker initialization code from a pickle file and call it."""
    with open(filename, mode="rb") as handle:
        func, *args = pickle.load(handle)
    func(*args)


class ProcessPool:
    """Define a process pool that handles the Windows platform specially."""

    def __init__(
        self,
        processes: Optional[int] = None,
        initializer: Optional[Callable] = None,
        initargs: Tuple = (),
        maxtasksperchild: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize a process pool.

        Add a thin layer on top of the `multiprocessing.Pool` that, on Windows, passes
        initialization code to workers via a pickle file rather than directly. This is
        done to avoid a performance issue that exists on Windows. Please, also see the
        discussion [1_].

        References
        ----------
        .. [1] https://github.com/opencobra/cobrapy/issues/997

        """
        super().__init__(**kwargs)
        self._filename = None
        if initializer is not None and system() == "Windows":
            descriptor, self._filename = mkstemp(suffix=".pkl")
            # We use the file descriptor to the open file returned by `mkstemp` to
            # ensure that the resource is closed and can later be removed. Otherwise
            # Windows will cause a `PermissionError`.
            with os.fdopen(descriptor, mode="wb") as handle:
                pickle.dump((initializer,) + initargs, handle)
            initializer = _init_win_worker
            initargs = (self._filename,)
        self._pool = multiprocessing.Pool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
        )

    def __getattr__(self, name: str, **kwargs) -> Any:
        """Defer attribute access to the pool instance."""
        return getattr(self._pool, name, **kwargs)

    def __enter__(self) -> "ProcessPool":
        """Enable context management."""
        self._pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Clean up resources when leaving a context."""
        # The `multiprocessing.Pool.__exit__` only terminates pool processes. For a
        # clean exit, we close the pool and join the pool processes first.
        try:
            self._pool.close()
            self._pool.join()
        finally:
            self._clean_up()
        result = self._pool.__exit__(exc_type, exc_val, exc_tb)
        return result

    def close(self) -> None:
        """
        Close the process pool.

        Prevent any more tasks from being submitted to the pool. Once all the tasks have
        been completed, the worker processes will exit.

        """
        try:
            self._pool.close()
        finally:
            self._clean_up()

    def _clean_up(self) -> None:
        """Remove the dump file if it exists."""
        if self._filename is not None and Path(self._filename).exists():
            Path(self._filename).unlink()
"""Additional helper functions for the optlang solvers.

All functions integrate well with the context manager, meaning that
all operations defined here are automatically reverted when used in a
`with model:` block.

The functions defined here together with the existing model functions
should allow you to implement custom flux analysis methods with ease.

"""




# Used to avoid cyclic reference and enable third-party static type checkers to work
if TYPE_CHECKING:
    from cobra import Model, Reaction


CONS_VARS = Union[optlang.interface.Constraint, optlang.interface.Variable]

logger = logging.getLogger(__name__)

# Define all the solvers that are found in optlang.
solvers = {
    match.split("_interface")[0]: getattr(optlang, match)
    for match in dir(optlang)
    if "_interface" in match
}

# Defines all the QP solvers implemented in optlang.
qp_solvers = ["cplex", "gurobi", "osqp"]

# optlang solution statuses which still allow retrieving primal values
has_primals = [NUMERIC, FEASIBLE, INFEASIBLE, SUBOPTIMAL, ITERATION_LIMIT, TIME_LIMIT]


class Components(NamedTuple):
    """Define an object for adding absolute expressions."""

    variable: optlang.interface.Variable
    upper_constraint: optlang.interface.Constraint
    lower_constraint: optlang.interface.Constraint


def linear_reaction_coefficients(
    model: "Model", reactions: Optional[List["Reaction"]] = None
) -> Dict["Reaction", float]:
    """Retrieve coefficient for the reactions in a linear objective.

    Parameters
    ----------
    model : cobra.Model
        The cobra model defining the linear objective.
    reactions : list of cobra.Reaction, optional
        An optional list of the reactions to get the coefficients for.
        By default, all reactions are considered (default None).

    Returns
    -------
    dict
        A dictionary where the keys are the reaction objects and the values
        are the corresponding coefficient. Empty dictionary if there are no
        linear terms in the objective.

    """
    linear_coefficients = {}
    reactions = model.reactions if not reactions else reactions
    try:
        objective_expression = model.solver.objective.expression
        coefficients = objective_expression.as_coefficients_dict()
    except AttributeError:
        return linear_coefficients
    for rxn in reactions:
        forward_coefficient = coefficients.get(rxn.forward_variable, 0)
        reverse_coefficient = coefficients.get(rxn.reverse_variable, 0)
        if forward_coefficient != 0:
            if forward_coefficient == -reverse_coefficient:
                linear_coefficients[rxn] = float(forward_coefficient)
    return linear_coefficients


def _valid_atoms(model: "Model", expression: optlang.symbolics.Basic) -> bool:
    """Check whether a sympy expression references the correct variables.

    Parameters
    ----------
    model : cobra.Model
        The model in which to check for variables.
    expression : sympy.Basic
        A sympy expression.

    Returns
    -------
    bool
        True if all referenced variables are contained in model, False
        otherwise.

    """
    atoms = expression.atoms(optlang.interface.Variable)
    return all(a.problem is model.solver for a in atoms)


def set_objective(
    model: "Model",
    value: Union[
        optlang.interface.Objective,
        optlang.symbolics.Basic,
        Dict["Reaction", float],
    ],
    additive: bool = False,
) -> None:
    """Set the model objective.

    Parameters
    ----------
    model : cobra.Model
       The model to set the objective for.
    value : optlang.interface.Objective, optlang.symbolics.Basic, dict
        If the model objective is linear, then the value can be a new
        optlang.interface.Objective or a dictionary with linear
        coefficients where each key is a reaction and the corresponding
        value is the new coefficient (float).
        If the objective is non-linear and `additive` is True, then only
        values of class optlang.interface.Objective, are accepted.
    additive : bool
        If True, add the terms to the current objective, otherwise start with
        an empty objective.

    Raises
    ------
    ValueError
        If model objective is non-linear and the `value` is a dict.
    TypeError
        If the type of `value` is not one of the accepted ones.

    """
    interface = model.problem
    reverse_value = model.solver.objective.expression
    reverse_value = interface.Objective(
        reverse_value, direction=model.solver.objective.direction, sloppy=True
    )

    if isinstance(value, dict):
        if not model.objective.is_Linear:
            raise ValueError(
                "You can only update non-linear objectives additively using object of "
                f"class optlang.interface.Objective, not of {type(value)}"
            )

        if not additive:
            model.solver.objective = interface.Objective(
                Zero, direction=model.solver.objective.direction
            )
        for reaction, coef in value.items():
            model.solver.objective.set_linear_coefficients(
                {reaction.forward_variable: coef, reaction.reverse_variable: -coef}
            )

    elif isinstance(value, (Basic, optlang.interface.Objective)):
        if isinstance(value, Basic):
            value = interface.Objective(
                value, direction=model.solver.objective.direction, sloppy=False
            )
        # Check whether expression only uses variables from current model;
        # clone the objective if not, faster than cloning without checking
        if not _valid_atoms(model, value.expression):
            value = interface.Objective.clone(value, model=model.solver)

        if not additive:
            model.solver.objective = value
        else:
            model.solver.objective += value.expression
    else:
        raise TypeError(f"{value} is not a valid objective for {model.solver}.")

    context = get_context(model)
    if context:

        def reset():
            model.solver.objective = reverse_value
            model.solver.objective.direction = reverse_value.direction

        context(reset)


def interface_to_str(interface: Union[str, ModuleType]) -> str:
    """Give a string representation for an optlang interface.

    Parameters
    ----------
    interface : str, ModuleType
        Full name of the interface in optlang or cobra representation.
        For instance, 'optlang.glpk_interface' or 'optlang-glpk'.

    Returns
    -------
    str
       The name of the interface as a string.
    """
    if isinstance(interface, ModuleType):
        interface = interface.__name__
    return re.sub(r"optlang.|.interface", "", interface)


def get_solver_name(mip: bool = False, qp: bool = False) -> str:
    """Select a solver for a given optimization problem.

    Parameters
    ----------
    mip : bool
        True if the solver requires mixed integer linear programming capabilities.
    qp : bool
        True if the solver requires quadratic programming capabilities.

    Returns
    -------
    str
        The name of the feasible solver.

    Raises
    ------
    SolverNotFound
        If no suitable solver could be found.

    """
    if len(solvers) == 0:
        raise SolverNotFound("No solvers found.")
    # Those lists need to be updated as optlang implements more solvers
    mip_order = ["gurobi", "cplex", "glpk"]
    lp_order = ["glpk", "cplex", "gurobi"]
    qp_order = ["gurobi", "cplex", "osqp"]

    if mip is False and qp is False:
        for solver_name in lp_order:
            if solver_name in solvers:
                return solver_name
        # none of them are in the list order - so return the first one
        return list(solvers)[0]
    elif qp:  # mip does not yet matter for this determination
        for solver_name in qp_order:
            if solver_name in solvers:
                return solver_name
        raise SolverNotFound("No QP-capable solver found.")
    else:
        for solver_name in mip_order:
            if solver_name in solvers:
                return solver_name
    raise SolverNotFound("No MIP-capable solver found.")


def choose_solver(
    model: "Model", solver: Optional[str] = None, qp: bool = False
) -> ModuleType:
    """Choose a solver given a solver name and model.

    This will choose a solver compatible with the model and required
    capabilities. Also respects model.solver where it can.

    Parameters
    ----------
    model : cobra.Model
        The model for which to choose the solver.
    solver : str, optional
        The name of the solver to be used (default None).
    qp : boolean, optional
        True if the solver needs quadratic programming capabilities
        (default False).

    Returns
    -------
    optlang.interface
        Valid solver for the problem.

    Raises
    ------
    SolverNotFound
        If no suitable solver could be found.

    """
    if solver is None:
        solver = model.problem
    else:
        model.solver = solver

    # Check for QP, raise error if no QP solver found
    if qp and interface_to_str(solver) not in qp_solvers:
        solver = solvers[get_solver_name(qp=True)]

    return solver


def check_solver(obj):
    """Check whether the chosen solver is valid.

    Check whether chosen solver is valid and also warn when using
    a specialized solver. Will return the optlang interface for the
    requested solver.

    Parameters
    ----------
    obj : str or optlang.interface or optlang.interface.Model
        The chosen solver.

    Raises
    ------
    SolverNotFound
        If the solver is not valid.
    """
    not_valid_interface = SolverNotFound(
        f"{obj} is not a valid solver interface. Pick one from {', '.join(solvers)}."
    )
    if isinstance(obj, str):
        try:
            interface = solvers[interface_to_str(obj)]
        except KeyError:
            raise not_valid_interface
    elif isinstance(obj, ModuleType) and hasattr(obj, "Model"):
        interface = obj
    elif isinstance(obj, optlang.interface.Model):
        interface = obj.interface
    else:
        raise not_valid_interface

    if interface_to_str(interface) in ["osqp", "coinor_cbc"]:
        logger.warning(
            "OSQP and CBC are specialized solvers for quadratic programming (QP) and "
            "mixed-integer programming (MIP) problems and may not perform well on "
            "general LP problems. So unless you intend to solve a QP or MIP problem, "
            "we recommend to change the solver back to a general purpose solver "
            "like `model.solver = 'glpk'` for instance."
        )

    return interface


def add_cons_vars_to_problem(
    model: "Model",
    what: Union[List[CONS_VARS], Tuple[CONS_VARS], Components],
    **kwargs,
) -> None:
    """Add variables and constraints to a model's solver object.

    Useful for variables and constraints that can not be expressed with
    reactions and lower/upper bounds. It will integrate with the model's
    context manager in order to revert changes upon leaving the context.

    Parameters
    ----------
    model : cobra.Model
       The model to which to add the variables and constraints.
    what : list or tuple of optlang.interface.Variable or
           optlang.interface.Constraint
       The variables and constraints to add to the model.
    **kwargs : keyword arguments
       Keyword arguments passed to solver's add() method.

    """
    model.solver.add(what, **kwargs)

    context = get_context(model)
    if context:
        context(partial(model.solver.remove, what))


def remove_cons_vars_from_problem(
    model: "Model",
    what: Union[List[CONS_VARS], Tuple[CONS_VARS], Components],
) -> None:
    """Remove variables and constraints from a model's solver object.

    Useful to temporarily remove variables and constraints from a model's
    solver object.

    Parameters
    ----------
    model : cobra.Model
       The model from which to remove the variables and constraints.
    what : list or tuple of optlang.interface.Variable or
           optlang.interface.Constraint
       The variables and constraints to remove from the model.

    """
    model.solver.remove(what)

    context = get_context(model)
    if context:
        context(partial(model.solver.add, what))


def add_absolute_expression(
    model: "Model",
    expression: str,
    name: str = "abs_var",
    ub: Optional[float] = None,
    difference: float = 0.0,
    add: bool = True,
) -> Components:
    """Add the absolute value of an expression to the model.

    Also defines a variable for the absolute value that can be used in
    other objectives or constraints.

    Parameters
    ----------
    model : cobra.Model
       The model to which to add the absolute expression.
    expression : str
       Must be a valid symbolic expression within the model's solver object.
       The absolute value is applied automatically on the expression.
    name : str, optional
       The name of the newly created variable (default "abs_var").
    ub : positive float, optional
       The upper bound for the variable (default None).
    difference : positive float, optional
        The difference between the expression and the variable
        (default 0.0).
    add : bool, optional
        Whether to add the variable to the model at once (default True).

    Returns
    -------
    Components
        A named tuple with variable and two constraints (upper_constraint,
        lower_constraint) describing the new variable and the constraints
        that assign the absolute value of the expression to it.

    """
    variable = model.problem.Variable(name, lb=0, ub=ub)
    # The following constraints enforce variable > expression and
    # variable > -expression
    upper_constraint = model.problem.Constraint(
        expression - variable, ub=difference, name="abs_pos_" + name
    )
    lower_constraint = model.problem.Constraint(
        expression + variable, lb=difference, name="abs_neg_" + name
    )
    to_add = Components(variable, upper_constraint, lower_constraint)
    if add:
        add_cons_vars_to_problem(model, to_add)
    return to_add


def fix_objective_as_constraint(
    model: "Model",
    fraction: float = 1.0,
    bound: Optional[float] = None,
    name: str = "fixed_objective_{}",
) -> float:
    """Fix current objective as an additional constraint.

    When adding constraints to a model, such as done in pFBA which
    minimizes total flux, these constraints can become too powerful,
    resulting in solutions that satisfy optimality but sacrifices too
    much for the original objective function. To avoid that, we can fix
    the current objective value as a constraint to ignore solutions that
    give a lower (or higher depending on the optimization direction)
    objective value than the original model.

    When done with the model as a context, the modification to the
    objective will be reverted when exiting that context.

    Parameters
    ----------
    model : cobra.Model
        The model to operate on.
    fraction : float, optional
        The fraction of the optimum the objective is allowed to reach
        (default 1.0).
    bound : float, optional
        The bound to use instead of fraction of maximum optimal value.
        If not None, `fraction` is ignored (default None).
    name : str, optional
        Name of the objective. May contain one "{}" placeholder which is
        filled with the name of the old objective
        (default "fixed_objective_{}").

    Returns
    -------
    float
        The value of the optimized objective * fraction

    """
    fix_objective_name = name.format(model.objective.name)
    if fix_objective_name in model.constraints:
        model.solver.remove(fix_objective_name)
    if bound is None:
        bound = model.slim_optimize(error_value=None) * fraction
    if model.objective.direction == "max":
        ub, lb = None, bound
    else:
        ub, lb = bound, None
    constraint = model.problem.Constraint(
        model.objective.expression, name=fix_objective_name, ub=ub, lb=lb
    )
    add_cons_vars_to_problem(model, constraint, sloppy=True)
    return bound


def check_solver_status(status: str = None, raise_error: bool = False) -> None:
    """Perform standard checks on a solver's status.

    Parameters
    ----------
    status: str, optional
        The status string obtained from the solver (default None).
    raise_error: bool, optional
        If True, raise error or display warning if False (default False).

    Returns
    -------
    None

    Warns
    -----
    UserWarning
        If `status` is not optimal and `raise_error` is set to True.

    Raises
    ------
    OptimizationError
        If `status` is None or is not optimal and `raise_error` is set to
        True.

    """
    if status == OPTIMAL:
        return None
    elif (status in has_primals) and not raise_error:
        warn(f"Solver status is '{status}'.", UserWarning)
    elif status is None:
        raise OptimizationError(
            "Model is not optimized yet or solver context has been switched."
        )
    else:
        raise OptimizationError(f"Solver status is '{status}'.")


def assert_optimal(model: "Model", message: str = "Optimization failed") -> None:
    """Assert model solver status is optimal.

    Do nothing if model solver status is optimal, otherwise throw
    appropriate exception depending on the status.

    Parameters
    ----------
    model : cobra.Model
        The model to check the solver status for.
    message : str, optional
        Message for the exception if solver status is not optimal
        (default "Optimization failed").

    Returns
    -------
    None

    Raises
    ------
    OptimizationError
       If solver status is not optimal.

    """
    status = model.solver.status
    if status != OPTIMAL:
        exception_cls = OPTLANG_TO_EXCEPTIONS_DICT.get(status, OptimizationError)
        raise exception_cls(f"{message} ({status}).")


def add_lp_feasibility(model: "Model") -> None:
    """Add a new objective and variables to ensure a feasible solution.

    The optimized objective will be zero for a feasible solution and
    otherwise represent the distance from feasibility (please see [1]_
    for more information).

    Parameters
    ----------
    model : cobra.Model
        The model whose feasibility is to be tested.

    Returns
    -------
    None

    References
    ----------
    .. [1] Gomez, Jose A., Kai Höffner, and Paul I. Barton.
    “DFBAlab: A Fast and Reliable MATLAB Code for Dynamic Flux Balance
    Analysis.” BMC Bioinformatics 15, no. 1 (December 18, 2014): 409.
    https://doi.org/10.1186/s12859-014-0409-8.

    """

    obj_vars = []
    prob = model.problem
    for met in model.metabolites:
        s_plus = prob.Variable("s_plus_" + met.id, lb=0)
        s_minus = prob.Variable("s_minus_" + met.id, lb=0)

        model.add_cons_vars([s_plus, s_minus])
        model.constraints[met.id].set_linear_coefficients({s_plus: 1.0, s_minus: -1.0})
        obj_vars.append(s_plus)
        obj_vars.append(s_minus)

    model.objective = prob.Objective(Zero, sloppy=True, direction="min")
    model.objective.set_linear_coefficients({v: 1.0 for v in obj_vars})


def add_lexicographic_constraints(
    model: "Model",
    objectives: List["Reaction"],
    objective_direction: Union[str, List[str]] = "max",
) -> pd.Series:
    """Successively optimize separate targets in a specific order.

    For each objective, optimize the model and set the optimal value as a
    constraint. Proceed in the order of the objectives given. Due to the
    specific order this is called lexicographic FBA [1]_. This procedure
    is useful for returning unique solutions for a set of important
    fluxes. Typically this is applied to exchange fluxes.

    Parameters
    ----------
    model : cobra.Model
        The model to be optimized.
    objectives : list of cobra.Reaction
        A list of reactions (or objectives) in the model for which unique
        fluxes are to be determined.
    objective_direction : str or list of str, optional
        The desired objective direction for each reaction (if a list) or
        the objective direction to use for all reactions (default "max").

    Returns
    -------
    pandas.Series
        A pandas Series containing the optimized fluxes for each of the
        given reactions in `objectives`.

    References
    ----------
    .. [1] Gomez, Jose A., Kai Höffner, and Paul I. Barton.
    “DFBAlab: A Fast and Reliable MATLAB Code for Dynamic Flux Balance
    Analysis.” BMC Bioinformatics 15, no. 1 (December 18, 2014): 409.
    https://doi.org/10.1186/s12859-014-0409-8.

    """

    if type(objective_direction) is not list:
        objective_direction = [objective_direction] * len(objectives)

    constraints = []
    for rxn_id, obj_dir in zip(objectives, objective_direction):
        model.objective = model.reactions.get_by_id(rxn_id)
        model.objective_direction = obj_dir
        constraints.append(fix_objective_as_constraint(model))

    return pd.Series(constraints, index=objectives)
