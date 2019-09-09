from flask import Blueprint
from flask import request, render_template, jsonify
from flask_cors import cross_origin
import json
from jsonschema import validate, ValidationError, SchemaError
from pathlib import Path
from db import Session
from models import OptimizationTask
from config import OPTIMIZATION_DATA, JSON_ENDING

# Schema that has to be validated against the uploaded json
JSON_SCHEMA_UPLOAD = "./json_schema/schema_upload.json"


optimization_blueprint = Blueprint("optimization", __name__)

# Our main upload page where we put our json
@optimization_blueprint.route("/upload", methods=['GET', 'POST'])
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
        optimization_type = req_data["optimization_type"]
        optimization = req_data["optimization"]
        # data is also necessary and can be retrieved directly; data will be written as file, as it's a big chunk
        # of data and can easily be stored/loaded as json
        data = req_data["data"]

        # Create instance of task
        optimizationtask = OptimizationTask(
                                author=author,
                                project=project,
                                optimization_id=optimization_id,
                                optimization_type=optimization_type,
                                optimization=optimization
                            )

        # Where to store the data
        filepath = f"{OPTIMIZATION_DATA}{optimization_id}{JSON_ENDING}"

        # Future: Check if optimization id exists in table and decide what to do if so

         # Try adding file to folder and adding job to table
        try:
            # Open created filepath
            with open(filepath, "w") as f:
                # Write json to it
                json.dump(data, f)

            # Add task to table
            Session.add(optimizationtask)
            # Push it to the server
            Session.commit()
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
@optimization_blueprint.route("/optimization", methods=["GET"])
@cross_origin()
def show_optimizations():
    pass