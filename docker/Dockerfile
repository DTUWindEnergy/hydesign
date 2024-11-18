FROM continuumio/anaconda3:latest
MAINTAINER Mikkel Friis-Møller <mikf@dtu.dk>

# build: 
# docker build . -t dtuwindenergy/hydesign:0.0.12
# tag:
# docker tag dtuwindenergy/hydesign dtuwindenergy/hydesign:latest (if needed)
# docker tag dtuwindenergy/hydesign dtuwindenergy/hydesign:<version number>
# push:
# docker login
# docker push --all-tags dtuwindenergy/hydesign

RUN apt-get update && \
    apt-get install make && \
    apt-get install libgl1-mesa-glx -y && \
    apt-get install gcc gfortran -y
RUN apt-get -y install make pandoc graphviz

RUN /opt/conda/bin/conda create -y -n hydesign_env python=3.11
SHELL ["conda", "run", "-n", "hydesign_env", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/hydesign_env/bin:$PATH   

# RUN conda install -y -c conda-forge -n hydesign_env smt
# RUN conda install -y -c conda-forge -n hydesign_env finitediff
RUN conda install -y -c conda-forge -n hydesign_env h5py==3.7
