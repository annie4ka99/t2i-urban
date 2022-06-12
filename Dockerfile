FROM python:3.8

COPY . /t2i-urban-inference-api/

ENV VIRTUAL_ENV=/t2i-urban-inference-api/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -r /t2i-urban-inference-api/requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
RUN pip install mmsegmentation
RUN pip3 install git+https://github.com/openai/CLIP.git


WORKDIR /t2i-urban-inference-api/

EXPOSE 8080

CMD [ "python", "/t2i-urban-inference-api/server.py" ]