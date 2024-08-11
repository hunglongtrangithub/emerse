#FROM nvcr.io/nvidia/pytorch:24.06-py3
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#FROM nvidia/cuda:12.5.0-devel-ubuntu20.04
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04
#set evironment variable for python version
ENV PYTHON_VERSION=3.10


# Install Python and essential packages (pip )
RUN apt-get update && \apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# install git
#RUN apt-get update && apt-get -y install git


COPY ./run.py /app/run.py
COPY ./.env /app/.env
WORKDIR /app

RUN pip install pyinstaller
RUN pip install torch 
RUN pip install transformers
RUN pip install accelerate
RUN pip install Flask
RUN pip install waitress
RUN pip install boto3
RUN pip install python-dotenv
RUN pip install html2text
RUN pyinstaller --onefile /app/run.py


EXPOSE 5000
