FROM ubuntu:20.04

RUN apt update && apt install -y git nano python3-pip swig python-opengl xvfb xserver-xephyr fontconfig libc6 \
&& rm -rf /var/lib/apt/lists/*

# Setup folder structure
RUN mkdir /RLProject
COPY . /RLProject

# Setup dependencies
WORKDIR /RLProject
RUN pip install -r requirements_docker.txt
