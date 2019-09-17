import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
import json
import numpy as np
from copy import deepcopy
from uuid import uuid4
from typing import Dict, Union, Tuple, List

from .evolutionary_toolbox import GAToolbox
# The following imports come from our flask app
# We want to use the same database and we need the same models for our query
from db import Session
from models import OptimizationTask, CalculationTask
from config import OPTIMIZATION_DATA, JSON_ENDING, CALC_INPUT_EXT, CALC_OUTPUT_EXT
from config import OPTIMIZATION_START, CALCULATION_START, OPTIMIZATION_RUN, CALCULATION_FINISH


def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        # Read json from it into data
        return json.load(f)


class OptimizationManager:
    def __init__(self,
                 session: Session,
                 optimization_task: OptimizationTask,
                 calculation_task: CalculationTask,
                 debug: bool = False):
        self.session = session
        self.optimization_task = optimization_task
        self.calculation_task = calculation_task
        self.debug = debug
        self.debug_counter = 0

    def debug_manager(self):
        """ Function used for debugging. The main purpose is to let python exit and therefor let docker restart the
        container.

        Returns:
            None - internal counter is modified and at some point this will cause an exit

        """
        self.debug_counter += 1
        if (self.debug_counter % 12) == 0:
            exit()

    @staticmethod
    def create_unique_id() -> uuid4:
        """ Function to create a unique id for calculation jobs

        Returns:
             unique id (uuid4) - a unique id to identify a calculation job

        """
        return uuid4()

    @staticmethod
    def get_weights(data) -> Tuple[int, int]:
        """ Function used to extract objectives from the json that holds the whole optimization task

        Args:
            data (dictionary) - the data as hold by the uploaded json

        Returns:
             objectives (tuple of ints) - the objectives of each function as presented in the json
        """
        return tuple(data["functions"][fun]["objective"] for fun in data["functions"])

    def query_first_optimizationtask(self):
        """

        Returns:
            query - first optimization task in list

        """
        return self.session.query(model).first()

    def run(self):
        """ Function run is used to keep the manager working constantly. It manages the following tasks:
        # What are the priorities for the optimization manager?
        # 1. Check if any optimization is finished and send out response.
        # 2. Check if any population calculation is finished and sum it up to generate a new population.
        # 3. Generate a new/starting population.
        # 4. Generate new calculation jobs.

        Returns:
            None - output is managed over databases
        """
        if self.debug:
            self.debug_manager()

        while True:
            if self.debug:
                print("No job. Sleeping for 10 seconds.")
                sleep(10)

            new_opt_task = Session.query(OptimizationTask).\
                filter(OptimizationTask.optimization_state == OPTIMIZATION_START).first()
            # If it returns an actual task
            if new_opt_task:
                # Set the optimization type in optimization task table to "optimization_run"
                # This is to prevent the manager from understanding this task as not already being started
                new_opt_task.optimization_state = OPTIMIZATION_RUN
                Session.commit()

                optimization_id = new_opt_task.optimization_id

                opt_filepath = new_opt_task.opt_path
                data_filepath = new_opt_task.data_path

                optimization = load_json(opt_filepath)

                data = load_json(data_filepath)


                gatoolbox = GAToolbox(
                    eta=optimization["parameters"]["eta"],
                    bounds=optimization["parameters"]["bounds"],
                    indpb=optimization["parameters"]["bounds"],
                    cxpb=optimization["parameters"]["bounds"],
                    mutpb=optimization["parameters"]["bounds"],
                    weights=get_weights(data)
                )

                number_of_generations = optimization["parameters"]["ngen"]

                population_size = optimization["parameters"]["pop_size"]

                # ind_pars = data["individual"]

                population = gatoolbox.make_population(population_size)

                # Start optimizing one task by looping over generations
                for generation in range(1, number_of_generations + 1):
                    new_opt_task.current_gen = generation
                    Session.commit()

                    if generation > 1:
                        finished_calculations = Session.query(CalculationTask).\
                            filter(CalculationTask.generation == generation,
                                   CalculationTask.calculation_state == CALCULATION_FINISH)

                        individuals_output = [json.load(open(calculation.calcoutput_filepath, "r"))
                                              for calculation in finished_calculations]

                        population = gatoolbox.optimize_evolutionary(individuals=individuals_output)

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

            continue


if __name__ == '__main__':
    optimization_manager = OptimizationManager(
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask
    )
    optimization_manager.run()
