FROM nvidia/cuda:11.1-devel-ubuntu18.04

#RUN apt-get update && apt-get install ca-certificates -y
#RUN apt-get install -y apt-transport-https
RUN apt-get update -y && \
    apt-get install -y \
    python3-pip \
    git

RUN pip3 install --upgrade pip
RUN pip3 install \
  torch \
  pytorch-lightning \
  git+https://github.com/norse/norse                 

ADD norse-dvs.py /norse-dvs.py
ADD main_cscs.sh /main_cscs.sh

CMD main_cscs.sh
