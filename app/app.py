import psycopg2
import urllib3
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import json
from sqlalchemy import create_engine

# Database as defined in container "db"
DATABASE_URL = "postgresql+psycopg2://root:root@db:5432/optimization"
# https://stackoverflow.com/questions/15634092/connect-to-an-uri-in-postgres
result = urllib3.util.parse_url(DATABASE_URL)
user, password = result.auth.split(":")
database = result.path[1:]
host = result.hostname
port = result.port

# engine_params = f"postgresql+psycopg2://{db_user}:{db_password}@{host}:{port}/{db_name}"

engine = create_engine(DATABASE_URL)
conn = engine.raw_connection()


# db = psycopg2.connect(
#     database="optimization",
#     user="root",
#     password="root",
#     host="db"
# )

SCHEMA_SERVER_URL = 'https://schema.inowas.com'

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        return json.dumps({
            'status': 200
        })

    if request.method == 'GET':
        if request.content_type == "application/json":
            return json.dumps({
                'message': "test"
            })
        return render_template('upload.html')


if __name__ == '__main__':
    app.secret_key = '2349978342978342907889709154089438989043049835890'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True, host='0.0.0.0')
