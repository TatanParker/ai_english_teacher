FROM tiangolo/uvicorn-gunicorn:python3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git make curl

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# Copy files
COPY ./pyproject.toml ./poetry.lock /app/
COPY ./Makefile /app/
COPY ./app /app/app

# Install dependencies
RUN make install
