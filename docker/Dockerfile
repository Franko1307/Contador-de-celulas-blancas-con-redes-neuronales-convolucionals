FROM tensorflow/tensorflow:latest-devel-gpu-py3

WORKDIR /estacion-de-trabajo

ADD . /estacion-de-trabajo

RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev




