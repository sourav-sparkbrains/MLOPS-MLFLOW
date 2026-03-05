FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY artifacts/ artifacts/
COPY utils/inference.py utils/inference.py

EXPOSE 8002

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8002"]