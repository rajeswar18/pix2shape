FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update --fix-missing
RUN apt-get update -y
RUN apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion vim emacs
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.0.5-2+cuda8.0 \
         libnccl-dev=2.0.5-2+cuda8.0 \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda install pytorch=0.3.1
RUN /opt/conda/bin/pip install torchvision
RUN /opt/conda/bin/pip install \
    matplotlib \
    plotly \
    dominate \
    opencv-python \
    tqdm
RUN conda install scipy scikit-image scikit-learn
RUN pip install beautifulsoup4
RUN pip install lxml
# RUN conda install ffmpeg -c conda-forge
RUN conda install -c numba numba
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp36-cp36m-linux_x86_64.whl

RUN apt-get update && apt-get install -y \
    python-tk \
    eog \
    strace

# Install some python dependences I need in my project
RUN pip install \
    future \
    scipy numpy six cycler pytz python-dateutil backports.functools-lru-cache pyparsing olefile Pillow decorator networkx PyWavelets scikit-image pandas seaborn \
    pyqt5 \
    pyopengl \
    argcomplete \
    lmdb \
    ipdb \
    hyperdash \
    graphviz \
    tensorboardX

CMD ["/bin/bash"]
