FROM pytorch/pytorch

RUN apt update && apt install -y git nano python3-pip swig\
&& rm -rf /var/lib/apt/lists/*

# Setup folder structure
RUN mkdir /RLProject
# COPY src/ /RLProject

# Setup dependencies
WORKDIR /RLProject
RUN pip install -r requirements_docker.txt
