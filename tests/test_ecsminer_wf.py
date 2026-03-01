"""Unit tests for the ECSMiner-WF model.

Verifies that ECSMiner-WF produces the expected metrics on the standard
RandomRBF benchmark dataset from the README, guarding against
unintended regressions.
"""
import numpy as np
import pytest

from streamndr.metrics import MNew, FNew, ErrRate
from streamndr.model import ECSMinerWF

# Expected metric values (from README)
EXPECTED_MNEW = 0.6093
EXPECTED_FNEW = 0.2678
EXPECTED_ERRRATE = 0.3940

# Tolerance for floating-point comparison
TOL = 1e-4


@pytest.fixture(scope="module")
def ecsminer_wf_metrics(dataset):
    """Train ECSMiner-WF and compute metrics on the online stream."""
    clf = ECSMinerWF(
        K=50,
        min_examples_cluster=10,
        verbose=0,
        random_state=42,
        ensemble_size=7,
        init_algorithm="kmeans",
    )
    clf.learn_many(
        np.array(dataset["X_train"]), np.array(dataset["y_train"])
    )

    known_classes = dataset["known_classes"]
    m_new = MNew(known_classes)
    f_new = FNew(known_classes)
    err_rate = ErrRate(known_classes)

    for x, y_true in zip(dataset["X_test"], dataset["y_test"]):
        y_pred = clf.predict_one(x)
        if y_pred is not None:
            m_new.update(y_true, y_pred[0])
            f_new.update(y_true, y_pred[0])
            err_rate.update(y_true, y_pred[0])

    return {"m_new": m_new.get(), "f_new": f_new.get(), "err_rate": err_rate.get()}


def test_ecsminer_wf_m_new(ecsminer_wf_metrics):
    assert ecsminer_wf_metrics["m_new"] == pytest.approx(EXPECTED_MNEW, abs=TOL)


def test_ecsminer_wf_f_new(ecsminer_wf_metrics):
    assert ecsminer_wf_metrics["f_new"] == pytest.approx(EXPECTED_FNEW, abs=TOL)


def test_ecsminer_wf_err_rate(ecsminer_wf_metrics):
    assert ecsminer_wf_metrics["err_rate"] == pytest.approx(EXPECTED_ERRRATE, abs=TOL)
