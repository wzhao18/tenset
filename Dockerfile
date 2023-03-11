FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && \
    apt-get install -y \
        python3 \
        python3-dev \
        python3-setuptools \
        gcc \
        libtinfo-dev \
        zlib1g-dev \
        build-essential \
        cmake \
        libedit-dev \
        libxml2-dev \
        wget \
        lsb-release \
        software-properties-common
    
RUN cd /home && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 15

COPY . /home/tvm

RUN cd /home/tvm && \
    mkdir build && \
    cp config.cmake build && \
    cd build && \
    cmake .. && \
    make -j4

