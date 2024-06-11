FROM python:3.12-slim

# Setting the working directory in the container
WORKDIR /app


RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app app

CMD ["python", "app/main.py"]