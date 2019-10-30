FROM python:3.6-stretch

MAINTAINER Ralf Junghanns <ralf.junghanns@gmail.com>

RUN buildDeps="unzip wget g++ gfortran make" && \
    apt-get update && \
    apt-get install -y $buildDeps --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install required packages
# Numpy needed in more current version then 1.12.0!
RUN pip install numpy==1.17.2
RUN pip install requests
# Compile modflow-excecutables
RUN pip install https://github.com/modflowpy/pymake/zipball/master

COPY pyMake-scripts /scripts

WORKDIR /scripts
RUN for file in ./*; do python $file 2>/dev/null; done
RUN mv ./temp/* /usr/local/bin

COPY . /app

WORKDIR /app

# Install packages
RUN pip install -r ./requirements.txt

# Install flopydatamodel and adapters
RUN pip install git+https://github.com/inowas/flopyAdapter@master

RUN rm -rf /pymake-1.1
