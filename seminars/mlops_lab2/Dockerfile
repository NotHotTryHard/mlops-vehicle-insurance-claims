FROM python:3.10-slim

WORKDIR /lab2

RUN pip install --no-cache-dir numpy==2.2.6 scikit-learn==1.7.2

COPY lab2_cli.py /lab2/lab2_cli.py

ENTRYPOINT ["python", "/lab2/lab2_cli.py"]
