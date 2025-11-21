FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app ./app

EXPOSE 7999

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7999"]
