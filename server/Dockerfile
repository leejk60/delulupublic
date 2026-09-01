# delulucam cloud avatar server.
# Built on the maintained FasterLivePortrait TensorRT image, which already
# carries CUDA/TensorRT and the ONNX->TensorRT conversion scripts — see
# https://github.com/warmshao/FasterLivePortrait
FROM shaoguo/faster_liveportrait:v3

WORKDIR /root/FasterLivePortrait
RUN pip install --no-cache-dir websockets

COPY realtime_server.py /root/FasterLivePortrait/realtime_server.py

# DELULUCAM_SERVER_TOKEN must be set at `docker run` time (-e); the server
# refuses to accept sessions without it. Convert models to TensorRT once
# before first run: `docker exec <container> bash scripts/all_onnx2trt.sh`
ENV DELULUCAM_SERVER_TOKEN=""
EXPOSE 8765
CMD ["python", "realtime_server.py"]
