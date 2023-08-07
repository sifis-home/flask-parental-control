import platform
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file

from app import get_data, on_close, on_error, on_open, video_capture
from factory import get_model

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = "6d7f7b7ced093a8b3ef6399163da6ece"


def test_get_model():
    weight_file = get_file(
        "EfficientNetB3_224_weights.11-3.44.hdf5",
        pretrained_model,
        cache_subdir="pretrained_models",
        file_hash=modhash,
        cache_dir=str(Path(__file__).resolve().parent),
    )

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist(
        [f"model.model_name={model_name}", f"model.img_size={img_size}"]
    )

    model = get_model(cfg)

    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg",
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(
        features
    )
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(
        features
    )
    expected_model = Model(
        inputs=base_model.input, outputs=[pred_gender, pred_age]
    )

    # Compare the models using their string representations
    if str(model) == str(expected_model):
        print("Test Passed: The loaded model matches the expected model.")
    else:
        print(
            "Test Failed: The loaded model does not match the expected model."
        )
        print("Expected Model Summary:")
        expected_model.summary()
        print("\nActual Loaded Model Summary:")
        model.summary()


def test_get_data():
    analyzer_id = platform.node()
    result = get_data()
    expected_result = analyzer_id
    assert result == expected_result


def test_on_error():
    error = "WebSocket error occurred"

    with patch("builtins.print") as mock_print:
        on_error(None, error)

    mock_print.assert_called_once_with(error)


def test_on_close():
    close_status_code = 1000
    close_msg = "Connection closed"

    with patch("builtins.print") as mock_print:
        on_close(None, close_status_code, close_msg)

    mock_print.assert_called_once_with("### Connection closed ###")


def test_on_open():
    with patch("builtins.print") as mock_print:
        on_open(None)

    mock_print.assert_called_once_with("### Connection established ###")


class TestVideoCapture(unittest.TestCase):
    @patch("cv2.VideoCapture")
    def test_video_capture_context_manager(self, mock_capture):
        mock_instance = mock_capture.return_value
        with video_capture("path/to/video.mp4") as cap:
            self.assertEqual(cap, mock_instance)
            mock_capture.assert_called_once_with("path/to/video.mp4")

        mock_instance.release.assert_called_once()
