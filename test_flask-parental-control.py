import platform
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
from omegaconf import OmegaConf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file

from app import (
    get_data,
    on_close,
    on_error,
    on_open,
    video_capture,
    yield_images,
)
from factory import get_model, get_optimizer, get_scheduler

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = "6d7f7b7ced093a8b3ef6399163da6ece"


def test_yield_images():
    # Mock cv2.VideoCapture
    mocked_capture = Mock(spec=cv2.VideoCapture)
    mocked_capture.set.return_value = True
    mocked_capture.read.side_effect = [
        ("mock_ret1", "mock_img1"),
        ("mock_ret2", "mock_img2"),
        (False, None),
    ]

    # Mock video_capture context manager
    mocked_video_capture = Mock(return_value=mocked_capture)
    with patch("app.video_capture", mocked_video_capture):
        # Call the function with a mock video link
        mock_video_link = "Adult1.mp4"

        # Create a generator object by calling the function
        generator = yield_images(mock_video_link)

        # Retrieve frames from the generator
        result1 = next(generator)
        result2 = next(generator)

        # Check if the VideoCapture context manager was called with the expected arguments
        mocked_video_capture.assert_called_once_with(mock_video_link)

        # Check if cv2.VideoCapture methods were called with the expected arguments
        mocked_capture.set.assert_called_once_with(
            cv2.CAP_PROP_FRAME_WIDTH, 640
        )
        mocked_capture.set.assert_called_once_with(
            cv2.CAP_PROP_FRAME_HEIGHT, 480
        )
        mocked_capture.read.assert_has_calls([call(), call()])

        # Check if the yielded frames match the expected frames
        assert result1 == "mock_img1"
        assert result2 == "mock_img2"

        # Check if a RuntimeError is raised when no more frames are available
        with pytest.raises(RuntimeError, match="Failed to capture image"):
            next(generator)


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


def test_get_optimizer():
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
    optimizer_name = "adam"
    lr = 0.001
    epochs = 30
    batch_size = 32
    cfg = OmegaConf.from_dotlist(
        [
            f"model.model_name={model_name}",
            f"model.img_size={img_size}",
            f"train.optimizer_name={optimizer_name}",
            f"train.lr={lr}",
            f"train.epochs={epochs}",
            f"train.batch_size={batch_size}",
        ]
    )

    optimizer = get_optimizer(cfg)

    if cfg.train.optimizer_name == "adam":
        expected_optimizer = Adam(lr=cfg.train.lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")

    # Compare the optimizers using their string representations
    if str(optimizer) == str(expected_optimizer):
        print(
            "Test Passed: The loaded optimizer matches the expected optimizer."
        )
    else:
        print(
            "Test Failed: The loaded optimizer does not match the expected optimizer."
        )
        print("Expected Optimizer Summary:")
        expected_optimizer.get_config()  # Display optimizer configuration
        print("\nActual Loaded Optimizer Summary:")
        optimizer.get_config()  # Display optimizer configuration


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


def test_get_scheduler():
    # Create a mock configuration (you can customize this according to your needs)
    class MockConfig:
        def __init__(self, epochs, lr):
            self.train = MockTrainConfig(epochs, lr)

    class MockTrainConfig:
        def __init__(self, epochs, lr):
            self.epochs = epochs
            self.lr = lr

    # Create a mock configuration with desired parameters
    mock_cfg = MockConfig(epochs=100, lr=0.1)

    # Get the scheduler from the function
    scheduler = get_scheduler(mock_cfg)

    # Test different epoch indices and expected learning rates
    epoch_indices = [0, 25, 50, 75, 100]
    expected_lr_values = [0.1, 0.02, 0.004, 0.0008, 0.0008]  # Adjust as needed

    for epoch_idx, expected_lr in zip(epoch_indices, expected_lr_values):
        calculated_lr = scheduler(epoch_idx)
        calculated_lr = round(calculated_lr, 4)
        assert (
            calculated_lr == expected_lr
        ), f"For epoch {epoch_idx}: expected {expected_lr}, got {calculated_lr}"
