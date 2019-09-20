import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
import json
from pathlib import Path
import numpy as np
from copy import deepcopy
from functools import partial
from uuid import uuid4
from typing import Dict, Union, Tuple, List, Optional
from tqdm import tqdm

from helper_functions import create_input_and_output_filepath, load_json, write_json
from .evolutionary_toolbox import GAToolbox
# The following imports come from our flask app
# We want to use the same database and we need the same models for our query
from db import Session
from models import OptimizationTask, CalculationTask
from config import CALC_INPUT_EXT, CALC_OUTPUT_EXT, OPTIMIZATION_START, CALCULATION_START, OPTIMIZATION_RUN, \
    CALCULATION_FINISH


def scalarize_solution(solution):
    return sum(solution)


class OptimizationManager:
    def __init__(self,
                 session,
                 gatoolbox,
                 optimization_task,
                 calculation_task,
                 debug: bool = False,
                 print_progress: bool = False):
        self.session = session
        self.gatoolbox = gatoolbox

        self.optimization_task = optimization_task
        self.calculation_task = calculation_task

        self.debug = debug
        self.debug_counter = 0
        self.print_progress = print_progress

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

    def query_first_starting_optimizationtask(self) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        return self.session.query(self.optimization_task)\
            .filter(self.optimization_task.optimization_state == OPTIMIZATION_START).first()

    def query_finished_calculationtasks(self,
                                        generation: str):

        return self.session.query(self.calculation_task).\
            filter(self.calculation_task.generation == generation,
                   self.calculation_task.calculation_state == CALCULATION_FINISH)

    def summarize_finished_calculationtasks(self,
                                            generation: str) -> List[dict]:
        finished_calculationtasks = self.query_finished_calculationtasks(generation=generation)

        return [load_json(calculation.calcoutput_filepath)
                for calculation in finished_calculationtasks]

    def await_generation_finished(self,
                                  optimization_id: str,
                                  generation: str,
                                  total_population: Optional[str] = None,
                                  print_progress: bool = False):
        if not total_population:
            total_population = Session.query(self.optimization_task).\
                filter(self.optimization_task.optimization_id == optimization_id).total_population

        if print_progress:
            pbar = tqdm(total=total_population)

        while True:
            current_population = Session.query(self.calculation_task).\
                filter(self.calculation_task.optimization_id == optimization_id,
                       self.calculation_task.generation == generation).count()

            if print_progress:
                pbar.update(current_population - pbar.n)

            if current_population == total_population:
                break

        if print_progress:
            pbar.close()

    def create_single_calculation_job(self,
                                      optimization_task: OptimizationTask,
                                      generation: str,
                                      individual: List[float],
                                      individual_id: str) -> None:
        calculation_parameters = {
            "ind_genes": individual
        }

        calculation_id = self.create_unique_id()

        calcinput_filepath, calcoutput_filepath = create_input_and_output_filepath(task_id=calculation_id,
                                                                                   extensions=[CALC_INPUT_EXT,
                                                                                               CALC_OUTPUT_EXT])

        new_calc_task = self.calculation_task(
            author=optimization_task.author,
            project=optimization_task.project,
            optimization_id=optimization_task.optimization_id,
            calculation_id=calculation_id,
            # Set state to start
            calculation_state=CALCULATION_START,
            generation=generation,
            individual_id=individual_id,
            data_filepath=optimization_task.data_filepath,
            calcinput_filepath=calcinput_filepath,
            calcoutput_filepath=calcoutput_filepath
        )

        write_json(obj=calculation_parameters,
                   filepath=calcinput_filepath)

        Session.add(new_calc_task)
        Session.commit()

    def create_new_calculation_jobs(self,
                                    optimization_task,
                                    generation: str,
                                    population):
        for i, individual in enumerate(population):
            self.create_single_calculation_job(optimization_task=optimization_task,
                                               generation=generation,
                                               individual=individual,
                                               individual_id=str(i+1))

    def linear_optimization_queue(self,
                                  optimization_task,
                                  individual,
                                  print_progress):
        generation = "linear_optimization"
        individual_id = "1"

        self.create_single_calculation_job(optimization_task=optimization_task,
                                           generation=generation,
                                           individual=individual,
                                           individual_id=individual_id)

        self.await_generation_finished(optimization_id=optimization_task.optimization_id,
                                       generation=generation,
                                       total_population=individual_id,
                                       print_progress=print_progress)

        solution_dict = self.summarize_finished_calculationtasks(generation=generation)

        solution = tuple(solution_dict["functions"][fun] for fun in solution_dict["functions"])

        return scalarize_solution(solution)

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

            new_optimization_task = self.query_first_starting_optimizationtask()

            if new_optimization_task:
                new_optimization_task.optimization_state = OPTIMIZATION_RUN
                Session.commit()

                optimization = load_json(new_optimization_task.opt_filepath)
                data = load_json(new_optimization_task.data_filepath)

                gatoolbox = self.gatoolbox(
                    eta=optimization["parameters"]["eta"],
                    bounds=optimization["parameters"]["bounds"],
                    indpb=optimization["parameters"]["indpb"],
                    cxpb=optimization["parameters"]["cxpb"],
                    mutpb=optimization["parameters"]["mutpb"],
                    weights=self.get_weights(data)
                )

                number_of_generations = optimization["parameters"]["ngen"]
                population_size = optimization["parameters"]["pop_size"]

                population = gatoolbox.make_population(population_size)

                for generation in range(1, number_of_generations + 1):
                    new_optimization_task.current_generation = str(generation)
                    new_optimization_task.current_population = 0
                    Session.commit()

                    if generation > 1:
                        individuals = self.summarize_finished_calculationtasks(generation=str(generation))

                        population = gatoolbox.optimize_evolutionary(individuals=individuals)

                    self.create_new_calculation_jobs(optimization_task=new_optimization_task,
                                                     generation=str(generation),
                                                     population=population)

                    self.await_generation_finished(optimization_id=new_optimization_task.optimization_id,
                                                   generation=str(generation),
                                                   print_progress=self.print_progress)

                individuals = self.summarize_finished_calculationtasks(generation=(number_of_generations + 1))

                population = self.gatoolbox.evaluate_finished_calculations(individuals=individuals)

                self.gatoolbox.select_best_individuals(population=population)

                solution = self.gatoolbox.select_first_of_hall_of_fame()

                solution = self.gatoolbox.optimize_linear(solution=solution,
                                                          function=partial(self.linear_optimization_queue,
                                                                           optimization_task=new_optimization_task,
                                                                           print_progress=self.print_progress))


if __name__ == '__main__':
    optimization_manager = OptimizationManager(
        session=Session,
        gatoolbox=GAToolbox,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask,
        debug=True,
        print_progress=True
    )

    optimization_manager.run()
