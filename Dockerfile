FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
 
WORKDIR /code
ENV PYTHON_VERSION 3.7.1
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
ENV TZ=Asia/Tokyo
ADD requirements.txt /code
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install tzdata
RUN apt-get -y install git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev && \
     apt-get -y install wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev && \
    git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    $PYENV_ROOT/plugins/python-build/install.sh && \
    /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT && \
    rm -rf $PYENV_ROOT && \
    pip install -r requirements.txt
