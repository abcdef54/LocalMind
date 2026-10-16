FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/backend /app/src/backend
COPY ./main.py /app/main.py
COPY ./alembic.ini /app/alembic.ini

EXPOSE 8000

CMD ["sh", "-c", "python main.py"]
