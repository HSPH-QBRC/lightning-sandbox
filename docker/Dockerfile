FROM ubuntu:noble

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    curl \
    python3-full

# Add the various python requirements and script files
ADD requirements.txt /opt/software/

# Install python dependencies
RUN /usr/bin/python3 -m venv /opt/software/venv
RUN opt/software/venv/bin/pip3 install -r /opt/software/requirements.txt
