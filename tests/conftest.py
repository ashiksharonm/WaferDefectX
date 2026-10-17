import numpy as np
import pytest

from wafer_inspect.features import extract_batch
from wafer_inspect.model import WaferModel, fit_calibrated
from wafer_inspect.synth import make_dataset


@pytest.fixture(scope="session")
def small_data():
    images, y, _ = make_dataset(150, seed=3)
    return images, y


@pytest.fixture(scope="session")
def model_file(tmp_path_factory, small_data):
    images, y = small_data
    path = tmp_path_factory.mktemp("m") / "wafer_model.joblib"
    WaferModel(fit_calibrated(extract_batch(images), y)).save(str(path))
    return str(path)


@pytest.fixture
def wafer_png():
    import cv2

    def _make(label="scratch", contrast=1.0, seed=0):
        from wafer_inspect.synth import make_wafer

        img = make_wafer(np.random.default_rng(seed), label, contrast)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        return buf.tobytes()

    return _make
