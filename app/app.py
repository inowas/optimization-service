from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import json
from jsonschema import validate, ValidationError, SchemaError
from models import db
# Import of the models
# https://www.compose.com/articles/using-postgresql-through-sqlalchemy/

JSON_SCHEMA_UPLOAD = "./json_schema/schema_upload.json"

DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route("/upload", methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # https://stackoverflow.com/questions/46136478/flask-upload-how-to-get-file-name
        if not request.files.get('file', None):
            error = "No file selected!"
            return render_template('upload.html', error=error)

        # Get json from request
        file_upload = request.files["file"]
        req_data = json.load(file_upload)

        with open(JSON_SCHEMA_UPLOAD, "r") as f:
            schema_upload = json.load(f)

        # Check the json against our json schema as provided in ./json_schema/schema_upload.json
        try:
            validate(instance=req_data, schema=schema_upload)
        except ValidationError as e:
            error = {
                "message": f"Validation failed. {e.message}",
                "code": e.validator,
                "schemapath": e.schema_path
            }

            return render_template('upload.html', error=error)

        except SchemaError as e:
            error = str(e)

            return render_template('upload.html', error=error)

        return jsonify(req_data)

    if request.method == 'GET':
        if request.content_type == "application/json":
            return json.dumps({
                'message': "test"
            })
        return render_template('upload.html')

    return render_template('upload.html')


if __name__ == "__main__":
    app.secret_key = '2349978342978342907889709154089438989043049835890'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, host='0.0.0.0')
