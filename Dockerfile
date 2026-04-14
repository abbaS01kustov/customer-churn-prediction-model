FROM python:3.8.12-slim

WORKDIR /app

COPY requirements.txt ./

# ADD THIS LINE
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./
COPY model_C=1.0.bin ./

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]