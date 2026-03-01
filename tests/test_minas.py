"""Unit tests for the MINAS model.

Verifies that MINAS produces the expected metrics on the standard
RandomRBF benchmark dataset from the README, guarding against
unintended regressions.
"""
import numpy as np
import pytest

from streamndr.metrics import ConfusionMatrixNovelty, MNew, FNew, ErrRate
from streamndr.model import Minas

# Expected metric values (from README)
EXPECTED_MNEW = 0.1715
EXPECTED_FNEW = 0.4011
EXPECTED_ERRRATE = 0.3680

# Tolerance for floating-point comparison
TOL = 1e-4


@pytest.fixture(scope="module")
def minas_metrics(dataset):
    """Train MINAS and compute metrics on the online stream."""
    clf = Minas(
        kini=100,
        cluster_algorithm="clustream",
        window_size=600,
        threshold_strategy=1,
        threshold_factor=1.1,
        min_short_mem_trigger=100,
        min_examples_cluster=20,
        verbose=0,
        random_state=42,
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


def test_minas_m_new(minas_metrics):
    assert minas_metrics["m_new"] == pytest.approx(EXPECTED_MNEW, abs=TOL)


def test_minas_f_new(minas_metrics):
    assert minas_metrics["f_new"] == pytest.approx(EXPECTED_FNEW, abs=TOL)


def test_minas_err_rate(minas_metrics):
    assert minas_metrics["err_rate"] == pytest.approx(EXPECTED_ERRRATE, abs=TOL)
