import datetime
import hashlib
import json
import platform
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import dlib
import numpy as np
import rel
import websocket
from flask import Flask, abort, json, request
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file

from factory import get_model

app = Flask(__name__)

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = "6d7f7b7ced093a8b3ef6399163da6ece"

margin = 0.4


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### Connection closed ###")


def on_open(ws):
    print("### Connection established ###")


def get_data():
    analyzer_id = platform.node()
    print(analyzer_id)

    return analyzer_id


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images(video_link):
    print(video_link)
    # capture video
    with video_capture(video_link) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


def yield_images_from_path(image_path):
    print(image_path)
    img = cv2.imread(str(image_path), 1)

    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r)))


@app.route(
    "/cam_object/<cam_link>/<Privacy_Parameter>/<requestor_id>/<requestor_type>/<request_id>"
)
def cam_object_recognition(
    cam_link, Privacy_Parameter, requestor_id, requestor_type, request_id
):
    analyzer_id = get_data()

    # Get current date and time
    now = datetime.datetime.now()

    # Generate a random hash using SHA-256 algorithm
    hash_object = hashlib.sha256()
    hash_object.update(bytes(str(now), "utf-8"))
    hash_value = hash_object.hexdigest()

    # Concatenate the time and the hash
    analysis_id = str(analyzer_id) + str(now) + hash_value

    # for age recognition
    cap = cv2.VideoCapture(cam_link)

    frame_id = 0

    weight_file = get_file(
        "EfficientNetB3_224_weights.11-3.44.hdf5",
        pretrained_model,
        cache_subdir="pretrained_models",
        file_hash=modhash,
        cache_dir=str(Path(__file__).resolve().parent),
    )

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]

    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist(
        [f"model.model_name={model_name}", f"model.img_size={img_size}"]
    )

    model = get_model(cfg)
    model.load_weights(weight_file)
    predicted_ages3 = []

    while True:
        # Read a frame from the video
        ret, img = cap.read()
        if not ret:
            break

        img = np.asarray(img)

        frame_id += 1
        print("frame_id: ", frame_id)

        input_img = cv2.cvtColor(
            cv2.GaussianBlur(img, (Privacy_Parameter, Privacy_Parameter), 0),
            cv2.COLOR_BGR2RGB,
        )
        private_img = cv2.cvtColor(
            cv2.GaussianBlur(img, (Privacy_Parameter, Privacy_Parameter), 0),
            cv2.COLOR_BGR2RGB,
        )
        # print(private_img.shape)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = (
                    d.left(),
                    d.top(),
                    d.right() + 1,
                    d.bottom() + 1,
                    d.width(),
                    d.height(),
                )
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(private_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(
                    private_img[yw1 : yw2 + 1, xw1 : xw2 + 1],
                    (img_size, img_size),
                )

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            predicted_ages2 = []

            for apparent_age in predicted_ages:
                if int(apparent_age) < 3:
                    age_text = "Toddler"
                elif int(apparent_age) >= 3 and int(apparent_age) <= 12:
                    age_text = "Child"
                elif int(apparent_age) >= 13 and int(apparent_age) <= 19:
                    age_text = "Teen"
                elif int(apparent_age) >= 20 and int(apparent_age) <= 60:
                    age_text = "Adult"
                elif int(apparent_age) > 60:
                    age_text = "Senior"
                predicted_ages2.append(age_text)

            print(predicted_ages2)
        predicted_ages3.append(predicted_ages2)
        print(len(predicted_ages3))

        ws_req = {
            "RequestPostTopicUUID": {
                "topic_name": "SIFIS:Privacy_Aware_Parental_Control_Frame_Results",
                "topic_uuid": "Parental_Control_Frame_Results",
                "value": {
                    "description": "Parental Control Frame Results",
                    "requestor_id": str(requestor_id),
                    "requestor_type": str(requestor_type),
                    "request_id": str(request_id),
                    "analyzer_id": str(analyzer_id),
                    "analysis_id": str(analysis_id),
                    "Type": "CAM",
                    "file_name": "Empty",
                    "Privacy_Parameter": int(Privacy_Parameter),
                    "Frame": int(frame_id),
                    "Ages": predicted_ages3,
                    "length": int(len(predicted_ages3)),
                },
            }
        }
        ws.send(json.dumps(ws_req))

    ws_req_final = {
        "RequestPostTopicUUID": {
            "topic_name": "SIFIS:Privacy_Aware_Parental_Control_Results",
            "topic_uuid": "Parental_Control_Results",
            "value": {
                "description": "Parental Control Results",
                "requestor_id": str(requestor_id),
                "requestor_type": str(requestor_type),
                "request_id": str(request_id),
                "analyzer_id": str(analyzer_id),
                "analysis_id": str(analysis_id),
                "Type": "CAM",
                "file_name": "Empty",
                "Privacy_Parameter": int(Privacy_Parameter),
                "Frames Count": int(frame_id),
                "Ages": predicted_ages3,
                "length": int(len(predicted_ages3)),
            },
        }
    }
    ws.send(json.dumps(ws_req_final))

    return ws_req_final


@app.route(
    "/file_estimation/<file_name>/<Privacy_Parameter>/<requestor_id>/<requestor_type>/<request_id>",
    methods=["POST"],
)
def handler(
    file_name, Privacy_Parameter, requestor_id, requestor_type, request_id
):
    analyzer_id = get_data()

    # Get current date and time
    now = datetime.datetime.now()

    # Generate a random hash using SHA-256 algorithm
    hash_object = hashlib.sha256()
    hash_object.update(bytes(str(now), "utf-8"))
    hash_value = hash_object.hexdigest()

    # Concatenate the time and the hash
    analysis_id = str(analyzer_id) + str(now) + hash_value

    Privacy_Parameter = int(Privacy_Parameter)
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    # Loop over every file that the user submitted.
    for filename, handle in request.files.items():
        # Create a temporary file.
        # The location of the temporary file is available in `temp.name`.
        temp = NamedTemporaryFile()
        # Write the user's uploaded file to the temporary file.
        # The file will get deleted when it drops out of scope.
        handle.save(temp)

        video_link = temp.name

        cap = cv2.VideoCapture(video_link)
        frame_id = 0

        weight_file = get_file(
            "EfficientNetB3_224_weights.11-3.44.hdf5",
            pretrained_model,
            cache_subdir="pretrained_models",
            file_hash=modhash,
            cache_dir=str(Path(__file__).resolve().parent),
        )

        # for face detection
        detector = dlib.get_frontal_face_detector()

        # load model and weights
        model_name, img_size = Path(weight_file).stem.split("_")[:2]

        img_size = int(img_size)
        cfg = OmegaConf.from_dotlist(
            [f"model.model_name={model_name}", f"model.img_size={img_size}"]
        )

        model = get_model(cfg)
        model.load_weights(weight_file)
        predicted_ages3 = []

        while True:
            # Read a frame from the video
            ret, img = cap.read()
            if not ret:
                break

            img = np.asarray(img)

            frame_id += 1
            print("frame_id: ", frame_id)

            input_img = cv2.cvtColor(
                cv2.GaussianBlur(
                    img, (Privacy_Parameter, Privacy_Parameter), 0
                ),
                cv2.COLOR_BGR2RGB,
            )
            private_img = cv2.cvtColor(
                cv2.GaussianBlur(
                    img, (Privacy_Parameter, Privacy_Parameter), 0
                ),
                cv2.COLOR_BGR2RGB,
            )
            # print(private_img.shape)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = (
                        d.left(),
                        d.top(),
                        d.right() + 1,
                        d.bottom() + 1,
                        d.width(),
                        d.height(),
                    )
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(
                        private_img, (x1, y1), (x2, y2), (255, 0, 0), 2
                    )
                    # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(
                        private_img[yw1 : yw2 + 1, xw1 : xw2 + 1],
                        (img_size, img_size),
                    )

                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                predicted_ages2 = []

                for apparent_age in predicted_ages:
                    if int(apparent_age) < 3:
                        age_text = "Toddler"
                    elif int(apparent_age) >= 3 and int(apparent_age) <= 12:
                        age_text = "Child"
                    elif int(apparent_age) >= 13 and int(apparent_age) <= 19:
                        age_text = "Teen"
                    elif int(apparent_age) >= 20 and int(apparent_age) <= 60:
                        age_text = "Adult"
                    elif int(apparent_age) > 60:
                        age_text = "Senior"
                    predicted_ages2.append(age_text)

                print(predicted_ages2)
            predicted_ages3.append(predicted_ages2)
            print(len(predicted_ages3))

            ws_req = {
                "RequestPostTopicUUID": {
                    "topic_name": "SIFIS:Privacy_Aware_Parental_Control_Frame_Results",
                    "topic_uuid": "Parental_Control_Frame_Results",
                    "value": {
                        "description": "Parental Control Frame Results",
                        "requestor_id": str(requestor_id),
                        "requestor_type": str(requestor_type),
                        "request_id": str(request_id),
                        "analyzer_id": str(analyzer_id),
                        "analysis_id": str(analysis_id),
                        "Type": "File",
                        "file_name": str(file_name),
                        "Privacy_Parameter": int(Privacy_Parameter),
                        "Frame": int(frame_id),
                        "Ages": predicted_ages2,
                        "length": int(len(predicted_ages2)),
                    },
                }
            }
            ws.send(json.dumps(ws_req))

        ws_req_final = {
            "RequestPostTopicUUID": {
                "topic_name": "SIFIS:Privacy_Aware_Parental_Control_Results",
                "topic_uuid": "Parental_Control_Results",
                "value": {
                    "description": "Parental Control Results",
                    "requestor_id": str(requestor_id),
                    "requestor_type": str(requestor_type),
                    "request_id": str(request_id),
                    "analyzer_id": str(analyzer_id),
                    "analysis_id": str(analysis_id),
                    "Type": "File",
                    "file_name": str(file_name),
                    "Privacy_Parameter": int(Privacy_Parameter),
                    "Frames Count": int(frame_id),
                    "Ages": predicted_ages3,
                    "length": int(len(predicted_ages3)),
                },
            }
        }
        ws.send(json.dumps(ws_req_final))

    return ws_req_final


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        "ws://localhost:3000/ws",
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    app.run(host="0.0.0.0", port=6060)
