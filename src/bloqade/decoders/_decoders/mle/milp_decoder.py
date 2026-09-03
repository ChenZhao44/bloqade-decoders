from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseMLEDecoder

if TYPE_CHECKING:
    import pulp


class MILPDecoder(BaseMLEDecoder):
    """MLE decoder using mixed-integer programming via PuLP.

    Finds the most likely error pattern matching an observed syndrome
    by solving a mixed integer program. Supports multiple solver
    backends through PuLP, including HiGHS, CPLEX, COPT, and Gurobi.

    The PuLP model is built once at construction; decoding a shot only
    replaces the right-hand side of the detector constraints before
    re-solving.

    Does NOT support decomposed error models with separator targets.
    Use ``detector_error_model(decompose_errors=False)`` instead.

    Args:
        dem: The detector error model describing the error structure.
        solver: Name of a PuLP solver (e.g. ``"HiGHS"``, ``"CPLEX_PY"``,
            ``"COPT"``, ``"GUROBI"``) or a ``pulp.LpSolver`` instance.
            Defaults to ``"HiGHS"``.
        verbose: If True, print solver output.

    Examples:
        >>> from bloqade.decoders import MILPDecoder
        >>> import stim
        >>> dem = stim.DetectorErrorModel(
        ...     '''
        ...     error(0.02) D0 L0
        ...     error(0.1) D1 L0
        ...     '''
        ... )
        >>> milp_decoder = MILPDecoder(dem)
        >>> milp_decoder_cbc = MILPDecoder(dem, solver="PULP_CBC_CMD")
    """

    def _setup_solver(
        self,
        solver: str | pulp.LpSolver = "HiGHS",
        verbose: bool = False,
        **_kwargs: Any,
    ) -> None:
        try:
            import pulp
        except ImportError as e:
            raise ImportError(
                "The pulp package is required for MILPDecoder. "
                'You can install it via: pip install "pulp"'
            ) from e

        if isinstance(solver, str):
            self._solver = pulp.getSolver(solver, msg=verbose)
        else:
            self._solver = solver

    def _instantiate(self, verbose: bool = False, **kwargs: Any) -> None:
        super()._instantiate(verbose=verbose, **kwargs)
        self._build_model()

    def _build_model(self) -> None:
        """Build the persistent PuLP model (variables and constraints)."""
        import pulp

        self._prob = pulp.LpProblem("mip", pulp.LpMaximize)
        self._error_variables = [
            pulp.LpVariable("e" + str(i), cat=pulp.LpBinary)
            for i in range(len(self._weights))
        ]
        self._prob += pulp.lpSum(
            w * self._error_variables[i] for i, w in enumerate(self._weights)
        )

        self._detector_constraints: list[pulp.LpConstraint] = []
        for i, dv in enumerate(self._detector_vertices):
            detector_variable = pulp.LpVariable(
                "h" + str(i),
                lowBound=0,
                upBound=len(dv),
                cat=pulp.LpInteger,
            )
            self._prob += (
                pulp.lpSum(self._error_variables[j] for j in dv)
                - 2 * detector_variable
                == 0,
                "c" + str(i),
            )
            self._detector_constraints.append(self._prob.constraints["c" + str(i)])

        self._logical_variables: list[pulp.LpVariable] = []
        for obs_idx, observable_index in enumerate(self._observable_indices):
            logical_var = pulp.LpVariable("l" + str(obs_idx), cat=pulp.LpBinary)
            self._logical_variables.append(logical_var)
            certain_flip = int(self._certain_obs_flip[obs_idx])
            if len(observable_index) == 0:
                self._prob += (
                    logical_var == certain_flip,
                    "lfix" + str(obs_idx),
                )
                continue
            slack_var = pulp.LpVariable(
                "u" + str(obs_idx),
                lowBound=0,
                upBound=len(observable_index),
                cat=pulp.LpInteger,
            )
            self._prob += (
                certain_flip
                + pulp.lpSum(self._error_variables[j] for j in observable_index)
                - 2 * slack_var
                == logical_var,
                "lpar" + str(obs_idx),
            )

    def _set_syndrome(self, detector_shot: np.ndarray) -> None:
        """Replace the detector-constraint RHS with a new shot's syndrome."""
        shot = np.asarray(detector_shot, dtype=int) ^ self._certain_det_flip
        # PuLP stores `expr == rhs` as `expr - rhs`, so constant = -rhs.
        for constraint, bit in zip(self._detector_constraints, shot):
            constraint.constant = -int(bit)

    def _fresh_solver(self, verbose: bool | None = None) -> pulp.LpSolver:
        # PuLP solver objects cache solve state (e.g. the GUROBI API
        # backend), so each solve must get a fresh solver instance.
        solver = copy.deepcopy(self._solver)
        show = self._verbose if verbose is None else verbose
        if show and hasattr(solver, "msg"):
            solver.msg = True
        return solver

    def _decode_error(
        self, det_shots: np.ndarray, confidence: np.ndarray | None = None
    ) -> np.ndarray:
        import pulp

        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)

        for d, detector_shot in enumerate(det_shots):
            self._set_syndrome(detector_shot)
            self._prob.solve(self._fresh_solver())
            if pulp.LpStatus[self._prob.status] != "Optimal":
                if self._verbose:
                    print(
                        "Did not find optimal solution",
                        pulp.LpStatus[self._prob.status],
                    )
                if confidence is not None:
                    confidence[d] = 0.0
                continue
            errors[d, :] = np.round(
                np.array([pulp.value(e) for e in self._error_variables]), decimals=0
            ).astype(bool)
        return errors

    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> tuple[BaseMLEDecoder._ConfidenceSolveResult | None, bool]:
        import pulp

        self._set_syndrome(detector_shot)

        added_constraints: list[str] = []
        if forbidden_logical is not None:
            diff_variables: list[pulp.LpVariable] = []
            for obs_idx, forbidden_bit in enumerate(forbidden_logical.astype(int)):
                diff_var = pulp.LpVariable("d" + str(obs_idx), cat=pulp.LpBinary)
                diff_variables.append(diff_var)
                name = "ddiff" + str(obs_idx)
                if forbidden_bit:
                    self._prob += (
                        diff_var + self._logical_variables[obs_idx] == 1,
                        name,
                    )
                else:
                    self._prob += (
                        diff_var == self._logical_variables[obs_idx],
                        name,
                    )
                added_constraints.append(name)
            self._prob += (
                pulp.lpSum(diff_variables) >= 1,
                "logical_difference",
            )
            added_constraints.append("logical_difference")

        try:
            self._prob.solve(self._fresh_solver(verbose))
            status = pulp.LpStatus[self._prob.status]
            if status == "Infeasible" and forbidden_logical is not None:
                return None, True
            if status != "Optimal":
                if verbose:
                    print("Did not find optimal solution", status)
                return None, False

            error = np.round(
                np.array([pulp.value(var) for var in self._error_variables]),
                decimals=0,
            ).astype(bool)
            logical = np.round(
                np.array([pulp.value(var) for var in self._logical_variables]),
                decimals=0,
            ).astype(bool)
            objective_value = float(cast(float, pulp.value(self._prob.objective)))
            return (
                self._ConfidenceSolveResult(
                    error=error,
                    logical=logical,
                    objective=objective_value,
                ),
                True,
            )
        finally:
            for name in added_constraints:
                self._prob.constraints.pop(name, None)
