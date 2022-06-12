FROM python:3.8
#FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

#ENV TZ=Europe/Moscow
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#RUN apt-get update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y apt-utils python3 python3-pip ffmpeg libsm6 libxext6 python3-venv git libxtst-dev libpng++-dev libjpeg-dev
#RUN pip3 install virtualenv

COPY . /t2i-urban-inference-api/

ENV VIRTUAL_ENV=/t2i-urban-inference-api/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#COPY ./requirements.txt /i2i-urban-inference-api/requirements.txt

#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6 -y

#RUN pip3 install -U pip wheel cmake ninja
RUN pip install -r /t2i-urban-inference-api/requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
RUN pip install mmsegmentation
RUN pip3 install git+https://github.com/openai/CLIP.git
#RUN pip3 install cython --no-cache-dir

#COPY ./models.tar.gz /i2i-urban-inference-api/
#RUN tar -zxf /i2i-urban-inference-api/models.tar.gz --directory /i2i-urban-inference-api
#RUN rm /i2i-urban-inference-api/models.tar.gz

#COPY ./models /i2i-urban-inference-api/models
#COPY ./checkpoints /i2i-urban-inference-api/checkpoints
#COPY ./result /i2i-urban-inference-api/result
#COPY ./results /i2i-urban-inference-api/results
#COPY server.py /i2i-urban-inference-api/server.py
#COPY ./config.yml /i2i-urban-inference-api/config.yml


WORKDIR /t2i-urban-inference-api/

EXPOSE 8080

CMD [ "python", "/t2i-urban-inference-api/server.py" ]