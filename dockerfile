# Dockerfile

FROM python:3.12.0

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 9008

CMD ["python", "agent_client.py"]
