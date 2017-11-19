FROM python:2

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python==3.3.0.10
RUN pip install tensorflow==1.4.0

COPY . .

CMD ["python", "-u", "run_demo_server.py", "--checkpoint-path", "/app/east_icdar2015_resnet_v1_50_rbox", "--debug" ]
# CMD ["/bin/bash"]