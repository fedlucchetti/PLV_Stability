FROM tensorflow/tensorflow:latest-gpu
RUN pip3 install matplotlib tqdm scipy pandas  sklearn
RUN apt install -y python3-numpy
RUN mkdir -p /home/PLV_Stability
WORKDIR /home/PLV_Stability
