# Use Python 3.11 slim image as base
FROM python:3.11-buster

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy project files
COPY src/ ./src

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

WORKDIR /app/src

# Install dependencies
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Run the application
CMD ["poetry", "run", "python", "app.py"]