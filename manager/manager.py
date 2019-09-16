import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
import json
import numpy as np
from copy import deepcopy
from uuid import uuid4
from .evolutionary_toolbox import GAToolbox
# The following imports come from our flask app
# We want to use the same database and we need the same models for our query
from db import Session
from models import OptimizationTask, CalculationTask
from config import OPTIMIZATION_DATA, JSON_ENDING, CALC_INPUT_EXT, CALC_OUTPUT_EXT
from config import OPTIMIZATION_START, CALCULATION_START, OPTIMIZATION_RUN, CALCULATION_FINISH


def run():
    i = 1
    while True:
        current_opt_task = Session.query(OptimizationTask).first()
        if current_opt_task:
            print(f'Current job: {current_opt_task.optimization_id}. working on it')
        else:
            print("No job. Sleeping for 10 seconds.")
            sleep(10)
            continue

        # What are the priorities for the optimization manager?
        # 1. Check if any optimization is finished and send out response.
        # 2. Check if any population calculation is finished and sum it up to generate a new population.
        # 3. Generate a new/starting population.
        # 4. Generate new calculation jobs.

        # 4
        # Query for first optimization task in list where type is optimization_start
        new_opt_task = Session.query(OptimizationTask).\
            filter(OptimizationTask.optimization_state == OPTIMIZATION_START).first()
        # If it returns an actual task
        if new_opt_task:
            # Set the optimization type in optimization task table to "optimization_run"
            # This is to prevent the manager from understanding this task as not already being started
            new_opt_task.optimization_state = OPTIMIZATION_RUN
            Session.commit()

            optimization_id = new_opt_task.optimization_id

            # Get filepaths
            opt_filepath = new_opt_task.opt_path
            data_filepath = new_opt_task.data_path

            ### Potential Error with loading ###
            # Open created filepath
            with open(opt_filepath, "r") as f:
                # Read json from it into data
                optimization = json.load(f)

            with open(data_filepath, "r") as f:
                # Read json from it into data
                data = json.load(f)
            #####################################

            gatoolbox = GAToolbox(
                eta=optimization["parameters"]["eta"],
                bounds=optimization["parameters"]["bounds"],
                indpb=optimization["parameters"]["bounds"],
                cxpb=optimization["parameters"]["bounds"],
                mutpb=optimization["parameters"]["bounds"],
                weights=(data["functions"][fun]["objective"] for fun in data["functions"])
            )

            # Create calculation jobs depending on optimization
            # First we need the generation size to loop over
            ngen = optimization["parameters"]["ngen"]
            # Secondly we need the information of population size to know how many tasks to create
            pop_size = optimization["parameters"]["pop_size"]
            # Also in this case we need information about our individuals
            ind_pars = data["individual"]

            # Start optimizing one task by looping over generations
            for generation in range(1, ngen + 1):
                # Set generation in optimization_task to gen (we can still use the same object, it's connected to
                # Session)
                new_opt_task.current_gen = generation
                Session.commit()

                if generation == 1:
                    population = gatoolbox.make_population(pop_size)
                else:

                    population = gatoolbox.optimize_evolutionary()

                for ind_id, ind_genes in enumerate(population):
                    individual = ind_id + 1

                    # Our individual consists of certain parameters as defined in data
                    calculation_parameters = {
                        "ind_genes": ind_genes
                    }

                    # Generate a unique calculation_id
                    calculation_id = uuid4()

                    # Create a filepath to that id
                    calcinput_filepath = f"{OPTIMIZATION_DATA}{calculation_id}{CALC_INPUT_EXT}{JSON_ENDING}"
                    calcoutput_filepath = f"{OPTIMIZATION_DATA}{calculation_id}{CALC_OUTPUT_EXT}{JSON_ENDING}"

                    # Create a new calculation task with
                    # 1. Author, Project and Optimization Id from optimization task table
                    # 2. calculation type as "calculation_start" and calculation parameters as the stuff that holds
                    # calculation responsive parameters
                    new_calc_task = CalculationTask(
                        author=new_opt_task.author,
                        project=new_opt_task.project,
                        optimization_id=optimization_id,
                        calculation_id=calculation_id,
                        # Set state to start
                        calculation_state=CALCULATION_START,
                        generation=generation,
                        individual=individual,
                        data_filepath=data_filepath,
                        calcinput_filepath=calcinput_filepath,
                        calcoutput_filepath=calcoutput_filepath
                    )

                    # Write calculation parameters
                    with open(calcinput_filepath, "w") as f:
                        json.dump(calculation_parameters, f)

                    # Add calculation task to session
                    Session.add(new_calc_task)
                    Session.commit()

                while True:
                    current_calculation_finished = Session.query(CalculationTask). \
                        filter(CalculationTask.generation == generation,
                               CalculationTask.calculation_state == CALCULATION_FINISH).count()

                    if current_calculation_finished == pop_size:
                        break

        sleep(10)

        i += 1

        if (i % 12) == 0:
            exit()

        continue


if __name__ == '__main__':
    run()
