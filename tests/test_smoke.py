import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import age_lower_bound, lift_at_fraction, make_demo_frame, run


def test_preprocessing_and_lift():
    assert age_lower_bound("[20-30)") == 20
    assert age_lower_bound("[0-10)") == 0
    assert lift_at_fraction(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.25
    ) == 2.0


def test_training_smoke():
    result = run(make_demo_frame())
    assert set(result["model"]) == {
        "Logistic regression", "Random forest", "XGBoost"
    }
    assert result["roc_auc"].between(0, 1).all()
