from io import BytesIO
from shutil import rmtree
from pathlib import Path
import json
from jsonschema import ValidationError, SchemaError, Draft7Validator
import pandas as pd
import matplotlib.pyplot as plt

from flask import Blueprint
from flask import request, render_template, jsonify, redirect, abort
from flask import Response
from flask_cors import cross_origin
from db import Session
from models import OptimizationTask, OptimizationHistory

from helpers.functions import get_table_for_optimization_id, create_input_and_output_filepath, \
    write_json, get_schema_and_refresolver
from helpers.config import SCHEMA_INOWAS_OPTIMIZATION, OPTIMIZATION_RUN, \
    OPTIMIZATION_DATA, OPTIMIZATION_FOLDER, JSON_ENDING

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
            return render_template('upload.html', error="No file selected!")

        file_upload = request.files["file"]
        request_data = json.load(file_upload)

        optimization_schema, refresolver = get_schema_and_refresolver(SCHEMA_INOWAS_OPTIMIZATION)

        try:
            Draft7Validator(schema=optimization_schema, resolver=refresolver).validate(request_data)

        except ValidationError as e:
            error = {
                "message": f"Validation failed. {e.message}",
                "code": e.validator,
                "schemapath": e.schema_path
            }

            return render_template('upload.html', error=error)

        except SchemaError as e:
            return render_template('upload.html', error=str(e))

        author = request_data.get("author", "unknown")
        project = request_data.get("project", "unknown")
        optimization_id = request_data["optimization_id"]
        optimization_state = request_data.get("type", "optimization_start")  # expect optimization_stop, otherwise start

        method = request_data["optimization"]["parameters"]["method"]
        population_size = request_data["optimization"]["parameters"]["pop_size"]
        total_generation = request_data["optimization"]["parameters"]["ngen"]

        optimizationtask = OptimizationTask(
                                author=author,
                                project=project,
                                optimization_id=optimization_id,
                                optimization_type=method,
                                optimization_state=optimization_state,  # Input: "optimization_start"
                                total_population=population_size,
                                total_generation=total_generation,
                                solution=dict()
                            )

        try:
            Path(OPTIMIZATION_DATA, OPTIMIZATION_FOLDER, optimization_id).mkdir(parents=True)
        except FileExistsError:
            pass

        data_filepath = (Path(OPTIMIZATION_DATA) / OPTIMIZATION_FOLDER / optimization_id /
                         f"optimization{JSON_ENDING}")

        try:
            write_json(obj=request_data,
                       filepath=data_filepath)

            Session.add(optimizationtask)

            Session.commit()
        except (UnicodeDecodeError, IOError):
            rmtree(Path(OPTIMIZATION_DATA, optimization_id))
            # Path(opt_filepath).unlink()
            #
            Path(data_filepath).unlink()

            Session.rollback()

            return abort(400, "Error: task couldn't be created!")

        return redirect(f"/optimization")  # /{optimization_id}

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

        optimization_tasks_df = optimization_tasks_df.drop(["data_filepath"], axis=1)

        return optimization_tasks_df.to_html()


# Optimization page with progress of running optimizations
@optimization_blueprint.route("/optimization/<optimization_id_>", methods=["GET"])
@cross_origin()
def show_single_optimization_progress(optimization_id_):
    if request.method == 'GET':

        optimization_task = Session.query(OptimizationTask).\
            filter(OptimizationTask.optimization_id == optimization_id_).first()

        if optimization_task:
            if optimization_task.optimization_state == OPTIMIZATION_RUN:
                optimization_id = optimization_task.optimization_id

                individual_oh = get_table_for_optimization_id(OptimizationHistory, optimization_id)

                optimization_progress = Session.query(individual_oh)

                if optimization_progress.first():
                    optimization_progress_df = pd.read_sql(optimization_progress.statement, Session.bind)
                    optimization_progress_df = optimization_progress_df.loc[
                        optimization_progress_df.generation <= (optimization_task.current_generation-1)]

                    df_with_all_generations = pd.DataFrame(
                        {"generation": range(1, optimization_task.total_generation + 1)})

                    optimization_progress_df = pd.merge(df_with_all_generations,
                                                        optimization_progress_df,
                                                        how="left")

                    output = BytesIO()

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.ylim((0, optimization_progress_df['scalar_fitness'].iloc[0]))

                    optimization_progress_df.plot(x="generation",
                                                  y="scalar_fitness",
                                                  logy=True,
                                                  legend=False,
                                                  ax=ax)

                    ax.set_title(f"Optimization ID: {optimization_id_}\n"
                                 f"Generation: {(optimization_task.current_generation-1)}/"
                                 f"{optimization_task.total_generation}")

                    ax.text(x=0.7,
                            y=0.9,
                            transform=ax.transAxes,
                            s=f"Current best fitness: {optimization_progress_df['scalar_fitness'].iloc[-1]}")

                    fig.savefig(output, format='png')

                    return Response(output.getvalue(), mimetype='image/png')

            return abort(404, f"Optimization with id {optimization_id_} isn't running. Progress graph not available!")

        return abort(404, f"Optimization with id {optimization_id_} does not exist.")
