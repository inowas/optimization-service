FROM python:3.6-stretch

MAINTAINER Ralf Junghanns <ralf.junghanns@gmail.com>

COPY . /app

WORKDIR /app

#RUN pip install numpy==1.12.0
#RUN pip install requests

# Install packages
RUN pip install -r ./requirements.txt

# Install flopydatamodel and adapters
RUN pip install git+https://github.com/inowas/flopyAdapter@feature/1770-creation-of-a-library-from-inowasflopyadapter
