import numpy as np
import pytest

from wafer_inspect import CLASSES
from wafer_inspect.active_learning import labels_needed, run
from wafer_inspect.features import FEATURE_NAMES, extract_features
from wafer_inspect.model import WaferModel, expected_calibration_error, normalized_entropy
from wafer_inspect.synth import make_dataset, make_wafer


def test_synth_is_deterministic_and_labelled():
    a, ya, ca = make_dataset(5, seed=1)
    b, yb, cb = make_dataset(5, seed=1)
    assert np.array_equal(a, b) and np.array_equal(ya, yb) and np.array_equal(ca, cb)
    assert a.shape == (5, 512, 512) and a.dtype == np.uint8
    with pytest.raises(ValueError):
        make_wafer(np.random.default_rng(0), "void")


def test_features_shape_and_shape_cues():
    rng = np.random.default_rng(0)
    scratch = extract_features(make_wafer(rng, "scratch", 1.0))
    particle = extract_features(make_wafer(rng, "particle", 1.0))
    good = extract_features(make_wafer(rng, "good", 1.0))
    assert scratch.shape == (len(FEATURE_NAMES),) and np.isfinite(scratch).all()
    i = FEATURE_NAMES.index
    assert scratch[i("elongation")] > particle[i("elongation")]
    assert scratch[i("z_max")] > good[i("z_max")]


def test_entropy_bounds():
    assert normalized_entropy(np.array([[1.0, 0, 0]]))[0] == pytest.approx(0.0, abs=1e-9)
    assert normalized_entropy(np.full((1, 3), 1 / 3))[0] == pytest.approx(1.0)


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    assert expected_calibration_error(perfect, y) == 0.0
    wrong = np.eye(3)[[1, 2, 0, 1]]
    assert expected_calibration_error(wrong, y) == pytest.approx(1.0)


def test_model_predictions(model_file, small_data):
    images, _ = small_data
    model = WaferModel.load(model_file)
    pred = model.predict_image(images[0])
    assert pred.label in CLASSES
    assert 1 / 3 <= pred.confidence <= 1.0
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= pred.uncertainty <= 1.0


def test_labels_needed():
    curve = {6: 0.5, 11: 0.85, 16: 0.92}
    assert labels_needed(curve, 0.9) == 16
    assert labels_needed(curve, 0.99) is None


def test_active_learning_runs_small():
    res = run(pool=60, test=40, seeds=1, batch=5, max_labels=16, target=0.5)
    assert set(res["strategies"]) == {"random", "uncertainty"}
    assert res["strategies"]["uncertainty"]["labels_needed_per_seed"][0] is not None


def test_train_writes_model_and_metrics(tmp_path):
    from wafer_inspect.train import run as train_run

    metrics = train_run(str(tmp_path), n_train=60, n_test=30)
    assert (tmp_path / "wafer_model.joblib").exists() and (tmp_path / "metrics.json").exists()
    assert 0.0 <= metrics["ece_calibrated"] <= 1.0 and "synthetic" in metrics["data"]
