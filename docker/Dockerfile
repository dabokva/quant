FROM tensorflow/tensorflow:2.21.0-gpu

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /tmp/requirements.txt

CMD ["/bin/bash"]