FROM python:3.11-buster

WORKDIR /app

COPY src/ ./src

EXPOSE 7860

WORKDIR /app/src

RUN pip install -r requirements.txt

CMD ["python", "app.py"]