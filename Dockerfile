#FROM nvcr.io/nvidia/pytorch:24.06-py3
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#FROM nvidia/cuda:12.5.0-devel-ubuntu20.04
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
#set evironment variable for python version
ENV PYTHON_VERSION=3.11


# Install Python and essential packages (pip )
RUN apt-get update && apt-get install -y --no-install-recommends python${PYTHON_VERSION}-dev python3-pip python${PYTHON_VERSION}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /app

COPY ./run.py ./pyproject.toml ./uv.lock ./.env ./

RUN uv sync --frozen
RUN ./.venv/bin/pyinstaller --onefile /app/run.py


EXPOSE 5000
