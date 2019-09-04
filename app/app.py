from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import json
from jsonschema import validate, ValidationError, SchemaError
from models import db, OptimizationTask
from pathlib import Path
# Import of the models
# https://www.compose.com/articles/using-postgresql-through-sqlalchemy/

# Schema that has to be validated against the uploaded json
JSON_SCHEMA_UPLOAD = "./json_schema/schema_upload.json"

# Folder for optimization data
OPTIMIZATION_DATA = "/optimization-data/"

# File ending for json
JSON_ENDING = ".json"

# Database to our optimization postgres
DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"

# Create a flask app
app = Flask(__name__)
# Configs for flask app
app.config["DEBUG"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Enable cross origin resource sharing
CORS(app)

# Add the created app to our database
# Basically our empty database with its models gets the information to which existing database to connect
db.init_app(app)

# Now we want to create all our tables/models (here optimization_tasks table)
# As we need to access the app we use app_context while creating the tables
with app.app_context():
    db.create_all()

# Our main upload page where we put our json
@app.route("/upload", methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    # This is our most important path
    if request.method == 'POST':
        # https://stackoverflow.com/questions/46136478/flask-upload-how-to-get-file-name
        # See if there's a file in our selection field
        if not request.files.get('file', None):
            error = "No file selected!"
            return render_template('upload.html', error=error)

        # Get json from request
        file_upload = request.files["file"]
        req_data = json.load(file_upload)

        # Load validation schema form json_schema
        with open(JSON_SCHEMA_UPLOAD, "r") as f:
            schema_upload = json.load(f)

        # Check the json against our json schema as provided in ./json_schema/schema_upload.json
        try:
            # Validate file
            validate(instance=req_data, schema=schema_upload)
        # In case there's an error with our data
        except ValidationError as e:
            # A message to show where there's a problem in the json file
            error = {
                "message": f"Validation failed. {e.message}",
                "code": e.validator,
                "schemapath": e.schema_path
            }

            return render_template('upload.html', error=error)
        # In case there's an error with our schema itself
        except SchemaError as e:
            # Print error
            error = str(e)

            return render_template('upload.html', error=error)

        # Now since no errors appeared the task will be added to the database
        # First we split our request data into our columns

        # Author, project are not necessary and will be replaced by standards if not given
        author = req_data.get("author", "Max Mustermann")
        project = req_data.get("project", "Standardprojekt")
        # optimization_id, optimization and type are necessary and can be retrieved directly
        optimization_id = req_data["optimization_id"]
        type = req_data["type"]
        optimization = req_data["optimization"]
        # data is also necessary and can be retrieved directly; data will be written as file, as it's a big chunk
        # of data and can easily be stored/loaded as json
        data = req_data["data"]

        # Create instance of task
        optimizationtask = OptimizationTask(
                                author=author,
                                project=project,
                                optimization_id=optimization_id,
                                opt_type=type,
                                optimization=optimization
                            )

        # Where to store the data
        filepath = f"{OPTIMIZATION_DATA}{optimization_id}{JSON_ENDING}"

         # Try adding file to folder and adding job to table
        try:
            # Open created filepath
            with open(filepath, "w") as f:
                # Write json to it
                json.dump(data, f)

            # With app_context
            with app.app_context():
                # Add task to table
                db.session.add(optimizationtask)
                # Push it to the server
                db.session.commit()
        except Exception:
            Path(filepath).unlink()

        return jsonify(req_data)

    if request.method == 'GET':
        if request.content_type == "application/json":
            return json.dumps({
                'message': "test"
            })
        return render_template('upload.html')


# Our optimization page where we can see the progress of running optimizations
@app.route("/optimization", methods=["GET"])
@cross_origin()
def show_optimizations():
    pass


if __name__ == "__main__":
    app.secret_key = '2349978342978342907889709154089438989043049835890'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, host='0.0.0.0')
