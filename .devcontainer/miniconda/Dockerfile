FROM continuumio/miniconda3:4.11.0
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
#RUN bash ~/miniconda.sh -b -f -p /usr/local/
RUN conda init
RUN conda update conda --yes
#eval "$(/miniconda/bin/conda shell.bash hook)"
RUN apt-get update && apt-get install -y build-essential sudo python3-opencv && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN conda install -c conda-forge opencv
RUN conda install ipykernel --yes
RUN conda install -c conda-forge jupyterlab
RUN conda install pandas --yes
RUN groupadd user && \
    useradd -m -g user -s /bin/bash user && \
    usermod -g user user && \
    echo "user:user" | chpasswd && \
    adduser user sudo
USER user
RUN conda init