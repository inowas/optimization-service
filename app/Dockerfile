FROM python:3.6-stretch

RUN apt-get update && apt-get install -y \
   postgresql-client

COPY . /app

WORKDIR /app

# Install the requirements
RUN pip install -r ./requirements.txt

CMD ["python", "app.py"]