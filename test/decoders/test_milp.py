"""Tests for the PuLP-based multi-solver MILP decoder."""

import stim
import numpy as np
import pytest

from bloqade.decoders import MILPDecoder

from .test_mle import regular_dem, regular_samples

# Solver backends MILPDecoder claims to support. Tests only run for solvers
# actually available in the environment (e.g. CPLEX/COPT need a license).
SUPPORTED_SOLVERS = [
    "HiGHS",
    "CPLEX_PY",
    "CPLEX_CMD",
    "COPT",
    "COPT_CMD",
    "GUROBI",
    "GUROBI_CMD",
]


def available_solvers() -> list[str]:
    import pulp

    available = set(pulp.listSolvers(onlyAvailable=True))
    return [name for name in SUPPORTED_SOLVERS if name in available]


SOLVERS = available_solvers()


@pytest.mark.parametrize("solver", SOLVERS)
def test_regular(solver):
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = MILPDecoder(dem, solver=solver)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


@pytest.mark.parametrize("solver", SOLVERS)
def test_hyper(solver):
    dem = stim.DetectorErrorModel("""
        error(0.1) D9 D0 D1 L0
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D3 D4
        error(0.1) D4 D5
        error(0.1) D5 D6
        error(0.1) D6 D7
        error(0.1) D7 D8
        error(0.1) D8 D9
        """)
    det_shots = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        bool,
    )
    obs_shots = np.array([[1], [0]], bool)
    decoder = MILPDecoder(dem, solver=solver)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


@pytest.mark.parametrize("solver", SOLVERS)
def test_no_error_syndrome(solver):
    dem = regular_dem()
    decoder = MILPDecoder(dem, solver=solver)
    det_shots = np.zeros((1, 10), dtype=bool)
    result = decoder.decode(det_shots)
    assert np.array_equal(result, np.array([[False]]))


@pytest.mark.parametrize("solver", SOLVERS)
def test_single_shot_decode(solver):
    dem = regular_dem()
    det_shots = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    decoder = MILPDecoder(dem, solver=solver)
    result = decoder.decode(det_shots)
    assert result.ndim == 1
    assert np.array_equal(result, np.array([True]))


def test_default_solver_is_available():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = MILPDecoder(dem)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_unknown_solver_rejected():
    dem = regular_dem()
    with pytest.raises(Exception):  # noqa: B017 - pulp raises its own error type
        MILPDecoder(dem, solver="NO_SUCH_SOLVER")


def test_separator_targets_rejected():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 ^ D1 L0
        """)
    with pytest.raises(ValueError, match="separator"):
        MILPDecoder(dem)


def test_milp_decoder_can_instantiate_without_training():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()

    decoder = MILPDecoder.instantiate(dem)
    result = decoder.decode(det_shots)

    np.testing.assert_array_equal(result, obs_shots)


def test_prob_zero_error_skipped():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.0) D1 D2 L0
        error(0.05) D0 D2
        """)
    decoder = MILPDecoder(dem)
    result = decoder.decode(np.array([[0, 1, 1]], dtype=bool))
    assert result.shape == (1, 1)


def test_prob_one_error_pre_applied():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(1.0) D1 D2 L0
        error(0.05) D0 D2
        """)
    decoder = MILPDecoder(dem)
    result = decoder.decode(np.array([[0, 1, 1]], dtype=bool))
    assert result.shape == (1, 1)
    assert result[0, 0]
