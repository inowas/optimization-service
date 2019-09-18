from flask import Blueprint
from flask import request, render_template, jsonify
from flask_cors import cross_origin
import json
from jsonschema import validate, ValidationError, SchemaError
from pathlib import Path

from helper_functions import create_input_and_output_filepath, load_json, write_json
from db import Session
from models import OptimizationTask
from config import OPT_EXT, DATA_EXT
from config import JSON_SCHEMA_UPLOAD

optimization_blueprint = Blueprint("optimization", __name__)

# Main page
@optimization_blueprint.route("/upload", methods=['GET', 'POST'])
@cross_origin()
def upload_file() -> jsonify:
    # This is our most important path
    if request.method == 'POST':
        # https://stackoverflow.com/questions/46136478/flask-upload-how-to-get-file-name
        # See if there's a file in our selection field
        if not request.files.get('file', None):
            error = "No file selected!"
            return render_template('upload.html', error=error)

        # No load from file, that's why no open!
        file_upload = request.files["file"]
        req_data = json.load(file_upload)

        schema_upload = load_json(JSON_SCHEMA_UPLOAD)

        try:
            validate(instance=req_data,
                     schema=schema_upload)

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

        author = req_data.get("author", "Max Mustermann")
        project = req_data.get("project", "Standardprojekt")
        optimization_id = req_data["optimization_id"]
        optimization_state = req_data["optimization_state"]
        optimization = req_data["optimization"]
        population_size = optimization["parameters"]["pop_size"]
        total_generation = optimization["parameters"]["ngen"]
        # data is also necessary and can be retrieved directly; data will be written as file, as it's a big chunk
        # of data and can easily be stored/loaded as json
        data = req_data["data"]

        # Where to store the optimization/data
        opt_filepath, data_filepath = create_input_and_output_filepath(task_id=optimization_id,
                                                                       extensions=[OPT_EXT, DATA_EXT])

        # Create instance of task
        optimizationtask = OptimizationTask(
                                author=author,
                                project=project,
                                optimization_id=optimization_id,
                                optimization_state=optimization_state,  # Input: "optimization_start"
                                total_population=population_size,
                                total_generation=total_generation,
                                opt_filepath=opt_filepath,
                                data_filepath=data_filepath
                            )

        # Future: Check if optimization id exists in table and decide what to do if so

         # Try adding file to folder and adding job to table
        try:
            write_json(obj=optimization,
                       filepath=opt_filepath)
            write_json(obj=data,
                       filepath=data_filepath)

            Session.add(optimizationtask)
            Session.commit()
        except Exception:
            Path(opt_filepath).unlink()
            Path(data_filepath).unlink()

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
