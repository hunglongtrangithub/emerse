#FROM nvcr.io/nvidia/pytorch:24.06-py3
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#FROM nvidia/cuda:12.5.0-devel-ubuntu20.04
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
#set evironment variable for python version
ENV PYTHON_VERSION=3.11


# Install Python and essential packages (pip )
RUN apt-get update &&  python${PYTHON_VERSION}-dev python3-pipapt-get install -y --no-install-recommends python${PYTHON_VERSION}

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the installer (version 0.4.1)
ADD https://astral.sh/uv/0.4.1/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.cargo/bin/:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# install git
#RUN apt-get update && apt-get -y install git


COPY ./run.py /app/run.py
COPY ./.env /app/.env
COPY ./uv.lock /app/uv.lock
WORKDIR /app

RUN RUN uv sync --frozen\
  && pyinstaller --onefile /app/run.py


EXPOSE 5000
