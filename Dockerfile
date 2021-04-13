FROM python:3.6

ARG PROG=trainer

WORKDIR /app
ADD . .

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    supervisor \
    curl \
    nginx && \
    apt-get -q clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /var/cache/apt/*.bin && \
    pip install -r requirements.txt

ENV VAULT_ADDR https://vault.example.com:8200
ENV PROG ${PROG}
CMD python $PROG.py
