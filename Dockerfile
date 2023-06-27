FROM python:3.7
RUN python -m pip install --upgrade pip
RUN pip install opencv-python-headless==4.5.3.56
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install poetry
RUN pip install ecc
RUN pip install websocket-client
RUN pip install rel

RUN git clone https://github.com/davisking/dlib
RUN pip install cmake
WORKDIR /dlib
RUN python setup.py install

#COPY . /app
#WORKDIR /app
COPY pyproject.toml /dlib
copy app.py /dlib

COPY pretrained_models /dlib
COPY EfficientNetB3_224_weights.11-3.44.hdf5 /dlib
COPY src /dlib
COPY factory.py /dlib
COPY generator.py /dlib
COPY utils.py /dlib
COPY config.yaml /dlib
COPY test.py /dlib

RUN poetry config virtualenvs.create false
RUN poetry install

# RUN pip install -r requirements.txt
EXPOSE 6060
ENTRYPOINT ["python"]
CMD ["app.py"]
