import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
import json
from sympy import lambdify, Symbol
from db import Session
from models import CalculationTask, OptimizationTask
from config import OPTIMIZATION_DATA, DATA_EXT, CALC_INPUT_EXT, CALC_OUTPUT_EXT, JSON_ENDING

from x_helper import g_mod

G_MOD_CONST = 10000

def run():
    while True:
        # Check if there's a new job in calculation tasks table
        new_calc_task = Session.query(CalculationTask). \
            filter(CalculationTask.calculation_type == "calculation_start").first()
        # If there actually is one
        if new_calc_task:
            # First important step is to set its type to calculation_run so that no other worker will do the same task
            new_calc_task.calculation_type = "calculation_run"
            Session.commit()

            # Now the calculation can happen
            # Get the optimization_id/calculation_id first
            optimization_id = new_calc_task.optimization_id
            calculation_id = new_calc_task.calculation_id

            # Create filepaths to optimization data and calculation input
            filepath_data = f"{OPTIMIZATION_DATA}{optimization_id}{DATA_EXT}{JSON_ENDING}"
            filepath_calc_input = f"{OPTIMIZATION_DATA}{calculation_id}{CALC_INPUT_EXT}{JSON_ENDING}"

            # Load our data which powers the calculation (basically our model)
            with open(filepath_data, "r") as f:
                data_input = json.load(f)

            # Load our calculation parameters that define what makes this calculation different from others
            with open(filepath_calc_input, "r") as f:
                calculation_parameters = json.load(f)

            # Generate x from individuals parameters
            x = g_mod(calculation_parameters["ind_genes"], G_MOD_CONST)

            # Here: Create our functions from the  text
            funs = data_input["functions"]
            # Our dictionary that will display our output data according to the functions we have calculated
            data_output = dict()
            data_output["ind_genes"] = calculation_parameters["ind_genes"]
            data_output["functions"] = dict()
            # Loop over functions
            for fun in funs.keys():
                # Create function from string
                x_symb = Symbol("x")
                f = lambdify(x_symb, funs[fun]["function"])
                # Calculate our "simulation" output
                data_output["functions"][fun] = f(x)

            # Filepath for data/simulation output
            filepath_calc_output = f"{OPTIMIZATION_DATA}{calculation_id}{CALC_OUTPUT_EXT}{JSON_ENDING}"

            # Write the results as json so that manager can read them and process them
            with open(filepath_calc_output, "w") as f:
                json.dump(data_output, f)

            # Set the calculation type of the task to calculation_finish
            new_calc_task.calculation_type = "calculation_finish"
            # Add 1 to optimization current runs
            opt_task = Session.query(OptimizationTask).\
                filter(OptimizationTask.optimization_id == optimization_id).first()
            opt_task.current_run += 1
            # Apply changes to database
            Session.commit()

            continue

        print('No jobs, sleeping for 1 minute')
        sleep(60)
        continue


if __name__ == '__main__':
    run()
