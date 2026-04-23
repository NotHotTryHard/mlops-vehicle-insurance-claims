FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy==2.2.6 scikit-learn==1.7.2

COPY . /app

ENTRYPOINT ["sh", "-c", "s=/app/seminars/predict.py; [ -f \"$s\" ] || s=/app/predict.py; exec python \"$s\" \"$@\"", "--"]
