import platform
import unittest
from unittest.mock import patch


from app import get_data, on_close, on_error, on_open


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
