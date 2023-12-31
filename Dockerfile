FROM python:3.10
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install poetry
RUN pip install ecc

RUN git clone https://github.com/davisking/dlib
RUN pip install cmake
WORKDIR /dlib
RUN python setup.py install

COPY pyproject.toml /dlib
copy app.py /dlib

COPY pretrained_models /dlib
COPY EfficientNetB3_224_weights.11-3.44.hdf5 /dlib
COPY src /dlib
COPY factory.py /dlib
COPY generator.py /dlib
COPY utils.py /dlib
COPY config.yaml /dlib
COPY config-3.py /dlib
COPY face.jpg /dlib

RUN poetry config virtualenvs.create false
RUN poetry install

EXPOSE 6060
ENTRYPOINT ["python"]
CMD ["app.py"]
