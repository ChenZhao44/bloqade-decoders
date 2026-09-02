from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..._mle_base import BaseMLEDecoder

if TYPE_CHECKING:
    import pulp


class MILPDecoder(BaseMLEDecoder):
    """MLE decoder using mixed-integer programming via PuLP.

    Finds the most likely error pattern matching an observed syndrome
    by solving a mixed integer program. Supports multiple solver
    backends through PuLP, including HiGHS, CPLEX, COPT, and Gurobi.

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

    def _decode_error(
        self, det_shots: np.ndarray, confidence: np.ndarray | None = None
    ) -> np.ndarray:
        import copy

        import pulp

        num_shots = det_shots.shape[0]
        num_errors = len(self._weights)
        errors = np.zeros([num_shots, num_errors], dtype=bool)

        weights = self._weights
        detector_vertices = self._detector_vertices
        # Pre-apply certain errors (prob=1.0) to the syndrome
        det_shots = det_shots.astype(int) ^ self._certain_det_flip

        for d, detector_shot in enumerate(det_shots):
            prob = pulp.LpProblem("mip", pulp.LpMaximize)
            error_variables = [
                pulp.LpVariable("e" + str(i), cat=pulp.LpBinary)
                for i in range(num_errors)
            ]
            prob += pulp.lpSum(w * error_variables[i] for i, w in enumerate(weights))

            for i, dv in enumerate(detector_vertices):
                detector_variable = pulp.LpVariable(
                    "h" + str(i),
                    lowBound=0,
                    upBound=len(dv),
                    cat=pulp.LpInteger,
                )
                prob += (
                    pulp.lpSum(error_variables[j] for j in dv) - 2 * detector_variable
                    == int(detector_shot[i]),
                    "c" + str(i),
                )

            # PuLP solver objects cache solve state (e.g. the GUROBI API
            # backend), so each problem must get a fresh solver instance.
            prob.solve(copy.deepcopy(self._solver))
            if pulp.LpStatus[prob.status] != "Optimal":
                if self._verbose:
                    print("Did not find optimal solution", pulp.LpStatus[prob.status])
                if confidence is not None:
                    confidence[d] = 0.0
                continue
            errors[d, :] = np.round(
                np.array([pulp.value(e) for e in error_variables]), decimals=0
            ).astype(bool)
        return errors

    def _solve_single_shot_for_confidence(
        self,
        detector_shot: np.ndarray,
        *,
        verbose: bool = False,
        forbidden_logical: np.ndarray | None = None,
    ) -> tuple[BaseMLEDecoder._ConfidenceSolveResult | None, bool]:
        import copy

        import pulp

        prob = pulp.LpProblem("mip", pulp.LpMaximize)
        weights = self._weights
        detector_vertices = self._detector_vertices
        observable_indices = self._observable_indices

        error_variables = [
            pulp.LpVariable("e" + str(i), cat=pulp.LpBinary)
            for i in range(len(weights))
        ]
        prob += pulp.lpSum(w * error_variables[i] for i, w in enumerate(weights))

        detector_shot = np.asarray(detector_shot, dtype=int) ^ self._certain_det_flip
        for i, detector_vertex in enumerate(detector_vertices):
            detector_variable = pulp.LpVariable(
                "h" + str(i),
                lowBound=0,
                upBound=len(detector_vertex),
                cat=pulp.LpInteger,
            )
            prob += (
                pulp.lpSum(error_variables[j] for j in detector_vertex)
                - 2 * detector_variable
                == int(detector_shot[i]),
                "c" + str(i),
            )

        logical_variables: list[pulp.LpVariable] = []
        for obs_idx, observable_index in enumerate(observable_indices):
            logical_var = pulp.LpVariable("l" + str(obs_idx), cat=pulp.LpBinary)
            logical_variables.append(logical_var)
            certain_flip = int(self._certain_obs_flip[obs_idx])
            if len(observable_index) == 0:
                prob += (
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
            prob += (
                certain_flip
                + pulp.lpSum(error_variables[j] for j in observable_index)
                - 2 * slack_var
                == logical_var,
                "lpar" + str(obs_idx),
            )

        if forbidden_logical is not None:
            diff_variables: list[pulp.LpVariable] = []
            for obs_idx, forbidden_bit in enumerate(forbidden_logical.astype(int)):
                diff_var = pulp.LpVariable("d" + str(obs_idx), cat=pulp.LpBinary)
                diff_variables.append(diff_var)
                if forbidden_bit:
                    prob += (
                        diff_var + logical_variables[obs_idx] == 1,
                        "ddiff" + str(obs_idx),
                    )
                else:
                    prob += (
                        diff_var == logical_variables[obs_idx],
                        "ddiff" + str(obs_idx),
                    )
            prob += (
                pulp.lpSum(diff_variables) >= 1,
                "logical_difference",
            )

        # PuLP solver objects cache solve state (e.g. the GUROBI API
        # backend), so each problem must get a fresh solver instance.
        solver = copy.deepcopy(self._solver)
        if verbose and hasattr(solver, "msg"):
            solver.msg = True
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if status == "Infeasible" and forbidden_logical is not None:
            return None, True
        if status != "Optimal":
            if verbose:
                print("Did not find optimal solution", status)
            return None, False

        error = np.round(
            np.array([pulp.value(var) for var in error_variables]), decimals=0
        ).astype(bool)
        logical = np.round(
            np.array([pulp.value(var) for var in logical_variables]), decimals=0
        ).astype(bool)
        objective_value = float(cast(float, pulp.value(prob.objective)))
        return (
            self._ConfidenceSolveResult(
                error=error,
                logical=logical,
                objective=objective_value,
            ),
            True,
        )
