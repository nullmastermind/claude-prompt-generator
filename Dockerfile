# Use Python 3.11 slim image as base
FROM python:3.11-buster

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY src/ ./src/

# Install dependencies
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860
# Change to src directory
WORKDIR /app/src

# Run the application
CMD ["poetry", "run", "python", "app.py"]