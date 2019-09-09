import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
import json
import numpy as np
from sqlalchemy.dialects.postgresql import UUID
# The following imports come from our flask app
# We want to use the same database and we need the same models for our query
from db import Session
from models import OptimizationTask, CalculationTask
from config import OPTIMIZATION_DATA, JSON_ENDING


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
            filter(OptimizationTask.optimization_type == "optimization_start").first()
        # If it returns an actual task
        if new_opt_task:
            # Get the optimization_id
            task_opt_id = new_opt_task.optimization_id
            # Create a filepath for the json that holds the data
            filepath = f"{OPTIMIZATION_DATA}{task_opt_id}{JSON_ENDING}"

            ### Potential Error with loading ###
            # Open created filepath
            with open(filepath, "r") as f:
                # Read json from it into data
                data = json.load(f)
            #####################################

            # Create calculation jobs depending on optimization
            # First we need the information of population size to know how many tasks to create
            task_opt = new_opt_task.optimization
            task_opt_parameters = task_opt["parameters"]
            pop_size = task_opt_parameters["pop_size"]

            ind_pars = data["individual"]

            # Now we loop over population size to create individuals
            for ind in range(pop_size):
                # Our individual consists of certain parameters as defined in data
                calculation_parameters = {
                    "ind_genes": [np.random.random(ind_pars["genes"]) *
                                  np.diff(ind_pars["boundaries"]) + np.min(ind_pars["boundaries"])]
                }

                # Create a new calculation task with
                # 1. Author, Project and Optimization Id from optimization task table
                # 2. calculation type as "calculation_start" and calculation parameters as the stuff that holds
                # calculation responsive parameters
                new_calc_task = CalculationTask(
                    author=new_opt_task.author,
                    project=new_opt_task.project,
                    optimization_id=new_opt_task.optimization_id,
                    calculation_type="calculation_start",
                    # If those parameters are to big, we should store them in files as well in the future!
                    calculation_parameters=calculation_parameters
                )

                # Add calculation task to session
                Session.add(new_calc_task)

            # Set the optimization type in optimization task table to "optimization_run"
            new_opt_task.optimization_type = "optimization_run"

            # Push the session with all the calculation tasks to the database
            # This should also change our optimization type as defined above (object relation)
            Session.commit()

        i += 1

        if (i % 12) == 0:
            exit()

        continue


if __name__ == '__main__':
    run()
