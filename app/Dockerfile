FROM python:3.6-stretch

RUN apt-get update && apt-get install -y \
   postgresql-client

COPY . /app

WORKDIR /app

RUN pip install numpy==1.12.0

# Install the requirements
RUN pip install -r ./requirements.txt

CMD [pytest, /tests]
CMD ["python", "app.py"]