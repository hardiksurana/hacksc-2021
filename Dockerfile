FROM python:3.8-slim-buster

COPY run.py /app/
COPY src /app/src/
COPY requirements.txt /pkg/

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r /pkg/requirements.txt

ARG app=run:app
ARG config=development
ARG host='0.0.0.0'
ARG port=80

ENV FLASK_APP=$app
ENV FLASK_ENV=$config
ENV FLASK_RUN_HOST=$host
ENV FLASK_RUN_PORT=$port

WORKDIR /app

ENTRYPOINT ["flask", "run"]