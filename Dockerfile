#FROM nvcr.io/nvidia/pytorch:24.06-py3
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#FROM nvidia/cuda:12.5.0-devel-ubuntu20.04
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
#set evironment variable for python version
ENV PYTHON_VERSION=3.11


# Install Python, pip, and essential packages
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python3-pip \
  python3-venv
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv from a Docker container or GitHub
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set Python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /app

COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-cache

COPY ./templates/ ./templates/
COPY ./src/ ./src/
COPY ./run.py ./

CMD ["uv", "run", "run.py"]

EXPOSE 5000
