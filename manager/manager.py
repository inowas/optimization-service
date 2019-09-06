import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
import json
# Both these imports come from our flask app
# We want to use the same database and we need the same model for our query
from db import Session
from models import OptimizationTask, CalculationTask
from config import OPTIMIZATION_DATA, JSON_ENDING


def run():
    while True:
        current_opt_task = Session.query(OptimizationTask).first()
        if current_opt_task:
            print(f'Current job: {current_opt_task.optimization_id}. working on it')
        else:
            print("No job. Sleeping for 10 seconds.")
        sleep(10)

        # What are the priorities for the optimization manager?
        # 1. Check if any optimization is finished and send out response.
        # 2. Check if any population calculation is finished and sum it up to generate a new population.
        # 3. Generate a new/starting population.
        # 4. Generate new calculation jobs.
        new_opt_task = Session.query(OptimizationTask).filter(OptimizationTask.type == "optimization_start").first()
        if new_opt_task:
            task_opt_id = new_opt_task.optimization_id
            task_opt = new_opt_task.optimization

            filepath = f"{OPTIMIZATION_DATA}{task_opt_id}{JSON_ENDING}"

            # Open created filepath
            with open(filepath, "r") as f:
                # Write json to it
                data = json.load(f)

            data

        continue


if __name__ == '__main__':
    run()
