FROM python:3.11-slim

WORKDIR /app

COPY requirements.docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Ensure the model directory is available inside the container
RUN mkdir -p /app/model /app/dataset

CMD ["python", "main.py"]
