FROM nvidia/cuda:11.1-devel-ubuntu18.04

#RUN apt-get update && apt-get install ca-certificates -y
#RUN apt-get install -y apt-transport-https
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    git && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install \
  torch \
  pytorch-lightning \
  git+https://github.com/norse/norse                 

ADD norse-dvs.py /norse-dvs.py

CMD python /norse-dvs.py --gpus=1
