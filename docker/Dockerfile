FROM ubuntu:jammy

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    curl \
    python3 \
    python3-pip

# Add the various python requirements and script files
ADD requirements.txt /opt/software/

# Install python dependencies
RUN pip3 install -r /opt/software/requirements.txt