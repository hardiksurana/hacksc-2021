FROM python:3.8-slim-buster

COPY run.py /app/
COPY src /app/src/
COPY requirements.txt /pkg/

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r /pkg/requirements.txt

EXPOSE 80

WORKDIR /app

ENTRYPOINT ["python", "/app/run.py"]