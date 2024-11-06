FROM python:3.11-slim

WORKDIR /usr/src/app

COPY src/ ./

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

RUN pip install --no-cache -r requirements.txt

CMD ["python", "app.py"]