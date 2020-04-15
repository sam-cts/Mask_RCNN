FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

RUN apt-get update
RUN mkdir /workspace
COPY . /workspace
RUN pip install 