FROM python:3.9-alpine

ADD . /src
WORKDIR /src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
