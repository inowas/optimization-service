from flask import Blueprint
from flask import request, render_template, jsonify, redirect, abort
from flask_cors import cross_origin
import json
from jsonschema import validate, ValidationError, SchemaError
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import HTML

from helper_functions import create_input_and_output_filepath, load_json, write_json
from db import Session
from models import OptimizationTask, OptimizationProgress
from config import OPT_EXT, DATA_EXT, JSON_SCHEMA_UPLOAD, OPTIMIZATION_RUN

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
        optimization_state = req_data["type"]
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
                                solution=dict(),
                                opt_filepath=opt_filepath,
                                data_filepath=data_filepath
                            )

        # Todo Check if optimization id exists in table and decide what to do if so

        try:
            write_json(obj=optimization,
                       filepath=opt_filepath)

            write_json(obj=data,
                       filepath=data_filepath)

            Session.add(optimizationtask)

            Session.commit()
        except (UnicodeDecodeError, IOError):
            Path(opt_filepath).unlink()

            Path(data_filepath).unlink()

            Session.rollback()

            # return render_html() creation error

        return redirect(f"/optimization/{optimization_id}")  # jsonify(req_data)

    if request.method == 'GET':
        if request.content_type == "application/json":
            return json.dumps({
                'message': "test"
            })
        return render_template('upload.html')


# Optimization page with progress of running optimizations
@optimization_blueprint.route("/optimization", methods=["GET"])
@cross_origin()
def show_all_optimizations():
    if request.method == 'GET':
        optimization_tasks = Session.query(OptimizationTask).statement

        optimization_tasks_df = pd.read_sql(optimization_tasks, Session.bind)

        optimization_tasks_df = optimization_tasks_df.drop(["opt_filepath", "data_filepath"], axis=1)

        return optimization_tasks_df.to_html()  # classes="table table-striped table-hover"


# Optimization page with progress of running optimizations
@optimization_blueprint.route("/optimization/<optimization_id_>", methods=["GET"])
@cross_origin()
def show_single_optimization_progress(optimization_id_):
    if request.method == 'GET':

        optimization_task = Session.query(OptimizationTask).\
            filter(OptimizationTask.optimization_id == optimization_id_).first()

        if optimization_task:
            if optimization_task.optimization_state == OPTIMIZATION_RUN:
                optimization_progress = Session.query(OptimizationProgress).\
                    filter(OptimizationProgress == optimization_id_)

                if optimization_progress.all():
                    optimization_progress_df = pd.read_sql(optimization_progress.statement, Session.bind)
                    optimization_progress_df = optimization_progress_df.loc[
                        optimization_progress_df.generation <= optimization_task.total_generation]

                    full_df = pd.DataFrame({"generation": range(1, optimization_task.total_generation + 1)})

                    optimization_progress_df = pd.merge(full_df, optimization_progress_df, how="left")

                    plt.figure()
                    optimization_progress_df.plot(x="generation", y="scalar_fitness")
                    plt.title(f"Optimization ID: {optimization_id_}"
                              f"Generation: {optimization_task.current_generation}/{optimization_task.total_generation}")

                    return plt

        return abort(404)

    pass
