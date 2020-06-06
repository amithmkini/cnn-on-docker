FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt update && apt -y install git curl
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -rf /tmp/miniconda.sh \
    && /opt/conda/bin/conda install -y python=3 \
    && /opt/conda/bin/conda update conda \
    && apt-get -qq -y remove curl \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && /opt/conda/bin/conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN conda install -y pytorch torchvision pylint matplotlib
