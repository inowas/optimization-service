import copy
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from sqlalchemy import cast, Date, or_
from typing import Dict, Union, Tuple, List, Optional
from helper_functions import create_input_and_output_filepath, load_json, write_json, get_table_for_optimization_id
from evolutionary_toolbox import EAToolbox
from db import Session, engine
from models import Base, OptimizationTask, CalculationTask, OptimizationHistory
from config import OPTIMIZATION_DATA
from config import CALC_INPUT_EXT, CALC_OUTPUT_EXT, OPTIMIZATION_START, CALCULATION_START, OPTIMIZATION_RUN, \
    CALCULATION_FINISH, OPTIMIZATION_FINISH, MAX_STORING_TIME_OPTIMIZATION_TASKS, OPTIMIZATION_ABORT, DATE_FORMAT


def scalarize_solution(solution):
    return sum(solution)


class OptimizationManager:
    def __init__(self,
                 session,
                 ea_toolbox,
                 optimization_task,
                 optimization_history,
                 calculation_task,
                 debug: bool = False,
                 print_progress: bool = False):
        # Db session and evolutionary algorithm toolbox
        self.session = session
        self.ea_toolbox = ea_toolbox

        # Preset tables
        self.ot = optimization_task
        self.oh = optimization_history
        self.ct = calculation_task

        # Temporary attributes (used for one optimization and then overwritten)
        self.current_optimization_id = None

        self.current_data = None
        self.current_variable_map = None

        self.current_eat = None
        self.current_oh = None
        self.current_ct = None

        # Others
        self.debug = debug
        self.debug_counter = 0
        self.print_progress = print_progress

    def reset_temporary_attributes(self):
        self.current_optimization_id = None

        self.current_data = None
        self.current_optimization_data = None
        self.current_variable_map = None

        self.current_eat = None
        self.current_oh = None
        self.current_ct = None

    # Author: Aybulat Fatkhutdinov; modified
    def apply_individual(self, individual):
        """Write individual values to variable template and return the filled template"""
        data = copy.deepcopy(self.current_data)
        for ind_value, keys in zip(individual, self.current_variable_map):
            if keys[1] == 'position':
                ind_value = int(ind_value)

            if keys[1] == 'concentration':
                for object_ in data["optimization"]["objects"]:
                    if object_['id'] == keys[0]:
                        object_[keys[1]][keys[2]][keys[3]]['result'] = ind_value
                        break
            else:
                for object_ in data["optimization"]["objects"]:
                    if object_['id'] == keys[0]:
                        object_[keys[1]][keys[2]]['result'] = ind_value
                        break

        return data

    # Authr: Aybulat Fatkhutdinov; modified
    def read_optimization_data(self,
                               optimization_data: dict = None):
        """
        Example of variables template, where values are fixed ones and Nones are optimized:

        Example of variables map and variables boundaries:
        var_map = [(0, flux, 0), (0, concentration, 0),(0, position, row),(0, position, col)]
        var_bounds = [(0, 10), (0, 1),(0, 30),(0, 30)]
        initial_values = [0, 0, 10, 10]
        """

        var_map = []
        var_bounds = []
        initial_values = []

        if optimization_data:
            optimization_data = optimization_data
        else:
            optimization_data = self.current_data["optimization"]

        for object_ in optimization_data["objects"]:
            for parameter, value in object_.items():
                if parameter == 'position':
                    for axis, axis_data in value.items():
                        if axis_data['min'] != axis_data['max']:
                            var_map.append((object_['id'], 'position', axis))
                            var_bounds.append((axis_data['min'], axis_data['max']))
                            initial_value = axis_data.get('result')
                            if initial_value is None:
                                initial_values.append(int((axis_data['max'] + axis_data['min']) / 2))
                            else:
                                initial_values.append(initial_value)
                            object_['position'][axis]['result'] = None
                        else:
                            object_['position'][axis]['result'] = axis_data['min']

                elif parameter == 'flux':
                    for period, period_data in value.items():
                        if period_data['min'] != period_data['max']:
                            var_map.append((object_['id'], 'flux', period))
                            var_bounds.append((period_data['min'], period_data['max']))
                            initial_value = period_data.get('result')
                            if initial_value is None:
                                initial_values.append((period_data['max'] + period_data['min']) / 2)
                            else:
                                initial_values.append(initial_value)
                            object_['flux'][period]['result'] = None
                        else:
                            object_['flux'][period]['result'] = period_data['min']

                elif parameter == 'concentration':
                    for period, period_data in value.items():
                        for component, component_data in period_data.items():
                            if component_data['min'] != component_data['max']:
                                var_map.append((object_['id'], 'concentration', period, component))
                                var_bounds.append((component_data['min'], component_data['max']))
                                initial_value = component_data.get('result')
                                if initial_value is None:
                                    initial_values.append((component_data['max'] + component_data['min']) / 2)
                                else:
                                    initial_values.append(initial_value)
                                object_[parameter][period][component]['result'] = None
                            else:
                                object_[parameter][period][component]['result'] = component_data['min']

        return var_map, var_bounds, initial_values

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
    def create_unique_id() -> str:
        """ Function to create a unique id for calculation jobs

        Returns:
             unique id (uuid4) - a unique id to identify a calculation job

        """
        return str(uuid4())

    def get_weights(self) -> tuple:
        """ Function used to extract objectives from the json that holds the whole optimization task

        Args:
            self (self) - optimization_data from optimization json

        Returns:
             objectives (tuple of ints) - the objectives of each function as presented in the json
        """
        return tuple(objective["weight"] for objective in self.current_data["optimization"]["objectives"])

    def linear_scalarization(self,
                             fitness: List[float]):
        scalar_fitness = 0
        for value, weight in zip(fitness, self.get_weights()):
            scalar_fitness += value * weight * -1
        return scalar_fitness

    def query_first_starting_optimization_task(self) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        return self.session.query(self.ot)\
            .filter(self.ot.optimization_state == OPTIMIZATION_START).first()

    def query_current_optimization_task(self) -> Session.query:
        """

        :param self:
        :return:
        """

        return self.session.query(self.ot).\
            filter(self.ot.optimization_id == self.current_optimization_id).first()

    def query_finished_calculation_tasks(self,
                                         generation: int) -> Session.query:
        """

        :param generation:
        :return:
        """

        return self.session.query(self.current_ct).\
            filter(self.current_ct.generation == generation,
                   self.current_ct.calculation_state == CALCULATION_FINISH).all()

    def summarize_finished_calculation_tasks(self,
                                             generation: int) -> Tuple[List[dict], list]:
        """

        :param generation:
        :return:
        """

        finished_calculation_tasks = self.query_finished_calculation_tasks(generation=generation)

        summarized_calculation_data = []
        summarized_fitness = []

        for calculation in finished_calculation_tasks:
            summarized_calculation_data.append(load_json(calculation.calcinput_filepath))
            summarized_fitness.append(load_json(calculation.calcoutput_filepath)["fitness"])

        return summarized_calculation_data, summarized_fitness

    def await_generation_finished(self,
                                  generation: int,
                                  total_population: Optional[int] = None):
        """

        :param generation:
        :param total_population:
        :return:
        """
        if not total_population:
            total_population = self.query_current_optimization_task().total_population

        while True:
            current_population = self.session.query(self.current_ct).\
                filter(self.current_ct.optimization_id == self.current_optimization_id,
                       self.current_ct.generation == generation,
                       self.current_ct.calculation_state == CALCULATION_FINISH).count()

            if current_population == total_population:
                break

    def create_single_calculation_job(self,
                                      individual: List[float],
                                      generation: int = None,
                                      individual_id: int = None) -> None:
        """

        :param individual:
        :param generation:
        :param individual_id:
        :return:
        """

        optimization_task = self.query_current_optimization_task()

        calculation_data = self.apply_individual(individual)

        calculation_id = self.create_unique_id()

        calcinput_filepath, calcoutput_filepath = create_input_and_output_filepath(
            folder=Path(OPTIMIZATION_DATA, self.current_optimization_id), task_id=calculation_id,
            file_types=[CALC_INPUT_EXT, CALC_OUTPUT_EXT])

        new_calc_task = self.current_ct(
            author=optimization_task.author,
            project=optimization_task.project,
            optimization_id=optimization_task.optimization_id,
            calculation_id=calculation_id,
            calculation_type=optimization_task.optimization_type,
            calculation_state=CALCULATION_START,  # Set state to start
            generation=generation,
            individual_id=individual_id,
            data_filepath=optimization_task.data_filepath,
            calcinput_filepath=calcinput_filepath,
            calcoutput_filepath=calcoutput_filepath
        )

        write_json(obj=calculation_data,
                   filepath=calcinput_filepath)

        # print(f"Generation {generation}: Wrote job json.")

        self.session.add(new_calc_task)
        self.session.commit()

        # print(f"Generation {generation}: Job commited to database.")

    def create_new_calculation_jobs(self,
                                    generation: int,
                                    population: List[List[float]]):
        """

        :param generation:
        :param population:
        :return:
        """
        for i, individual in enumerate(population):
            self.create_single_calculation_job(generation=generation,
                                               individual=individual,
                                               individual_id=i)

    def linear_optimization_queue(self,
                                  individual: List[float]):
        """

        :param individual:
        :return:
        """

        calculation_task = self.session.query(self.current_ct).last()

        if calculation_task:
            generation = calculation_task.generation + 1
        else:
            generation = 1

        individual_id = 0
        total_population = 1

        individual = list(individual)  # Mystic solver converts solution to a numpy array; we need a list

        self.create_single_calculation_job(generation=generation,
                                           individual=individual,
                                           individual_id=individual_id)

        self.await_generation_finished(generation=generation,
                                       total_population=total_population)

        _, summarized_fitness = self.summarize_finished_calculation_tasks(generation=generation)

        scalar_solution = self.linear_scalarization(summarized_fitness[0]["fitness"])

        # print(solution)

        optimization_task = self.query_current_optimization_task()

        optimization_task.scalar_fitness = scalar_solution
        self.session.commit()

        return scalar_solution

    def remove_optimization_and_calculation_data(self) -> None:
        optimization_task = self.query_current_optimization_task()

        Path(optimization_task.data_filepath).unlink()

        individual_ct = get_table_for_optimization_id(self.ct, self.current_optimization_id)

        calculations = self.session.query(individual_ct).all()

        calculation_files = [(calculation.calcinput_filepath,  calculation.calcoutput_filepath)
                             for calculation in calculations]

        for calcinput_file, calcoutput_file in calculation_files:
            Path(calcinput_file).unlink()
            Path(calcoutput_file).unlink()

    def remove_old_optimization_tasks_and_tables(self):
        now_date = datetime.now().date()

        optimization_tasks = self.session.query(self.ot)\
            .filter(or_(self.ot.optimization_state == OPTIMIZATION_FINISH,
                        self.ot.optimization_state == OPTIMIZATION_ABORT)).all()

        # old_optimization_tasks = [task
        #                           for task in optimization_tasks
        #                           if ((now_date - datetime.strptime(task.publishing_date, DATE_FORMAT)).days >
        #                               MAX_STORING_TIME_OPTIMIZATION_TASKS)]
        #
        # if old_optimization_tasks:
        #     for task in old_optimization_tasks:
        #         individual_ct = get_table_for_optimization_id(self.ct, self.current_optimization_id)
        #         individual_oh = get_table_for_optimization_id(self.oh, self.current_optimization_id)
        #
        #         Base.metadata.drop_all(tables=[individual_ct, individual_oh], bind=engine)
        #
        #         self.session.remove(task)
        #         self.session.commit()

    def manage_evolutionary_optimization(self):
        number_of_generations = self.current_data["optimization"]["parameters"]["ngen"]
        population_size = self.current_data["optimization"]["parameters"]["pop_size"]

        population = self.current_eat.make_population(population_size)

        for generation in range(number_of_generations):
            if self.debug:
                print(f"Generation: {generation}")

            optimization_task = self.query_current_optimization_task()

            optimization_task.current_generation = generation + 1  # Table counts to ten
            optimization_task.current_population = 0
            self.session.commit()

            if generation > 0:
                summarized_calculation_data, summarized_fitness = self.summarize_finished_calculation_tasks(
                    generation=(generation - 1))

                individuals = [self.read_optimization_data(single_calculation["optimization"])
                               for single_calculation in summarized_calculation_data]

                fitnesses = [self.linear_scalarization(single_fitness)
                             for single_fitness in summarized_fitness]

                if self.debug:
                    print(f"Individuals summarized.")

                population = self.current_eat.optimize_evolutionary(individuals=individuals, fitnesses=fitnesses)

                if self.debug:
                    print(f"Population developed.")

            self.create_new_calculation_jobs(generation=generation,
                                             population=population)

            if self.debug:
                print(f"New jobs created.")

            self.await_generation_finished(generation=generation)

            if self.debug:
                print(f"Jobs finished.")

            individuals = self.summarize_finished_calculation_tasks(generation=generation)

            if self.debug:
                print(f"Generation summarized.")

            population = self.current_eat.evaluate_finished_calculations(individuals=individuals)

            if self.debug:
                print(f"Generation evaluated.")

            population = self.current_eat.select_best_individuals(population=population)

            optimization_history = self.current_oh(
                author=optimization_task.author,
                project=optimization_task.project,
                optimization_id=self.current_optimization_id,
                generation=(generation + 1),
                scalar_fitness=self.linear_scalarization(
                    self.current_eat.select_nth_of_hall_of_fame(1)[0].fitness.values)
            )

            self.session.add(optimization_history)
            self.session.commit()

            if self.debug:
                print("Generation selected.")

        return self.current_eat.select_nth_of_hall_of_fame(self.current_data["optimization"]["number_of_solutions"])

    def manage_linear_optimization(self):
        # def custom_linear_optimization_queue(individual):
        #     return self.linear_optimization_queue(individual)

        solution = self.current_eat.optimize_linear(solution=self.current_data["optimization"]["result"],
                                                    function=self.linear_optimization_queue)

        if self.debug:
            print("Solution linear optimized.")
            print(f"Solution: {solution}")

        return solution

    def manage_any_optimization(self):
        optimization_task = self.session.query(self.ot)\
            .filter(self.ot.optimization_id == self.current_optimization_id).first()

        assert optimization_task.optimization_type in ["GA", "LO"], "Error: optimization_type is neither 'GA' nor 'LO'"

        if optimization_task.optimization_type == "GA":
            return self.manage_evolutionary_optimization()

        if optimization_task.optimization_type == "LO":
            return self.manage_linear_optimization()

    def run(self):
        """ Function run is used to keep the manager working constantly. It will work on one optimization only and
        fulfill the job which includes constantly creating jobs for one generation, then after calculation
        summarizing the results and creating new generations with new jobs and finally put the solution back in the
        table and set the optimization to be finished.

        Returns:
            None - output is managed over databases

        """

        while True:
            new_optimization_task = self.query_first_starting_optimization_task()

            if new_optimization_task:
                print(f"Working on task with id: {new_optimization_task.optimization_id}")

                self.current_optimization_id = new_optimization_task.optimization_id

                optimization_task = self.query_current_optimization_task()
                optimization_task.optimization_state = OPTIMIZATION_RUN
                self.session.commit()

                # Set temporary attributes
                self.current_data = load_json(new_optimization_task.data_filepath)

                variable_map, variable_bounds, _ = self.read_optimization_data()

                self.current_variable_map = variable_map

                # Set temporary attributes2
                self.current_eat = self.ea_toolbox(
                    bounds=variable_bounds,
                    weights=self.get_weights(),  # from self.optimization_data
                    parameters=self.current_data["optimization"]["parameters"]
                )
                self.current_oh = get_table_for_optimization_id(self.oh, self.current_optimization_id)
                self.current_ct = get_table_for_optimization_id(self.ct, self.current_optimization_id)

                Base.metadata.create_all(bind=engine,
                                         tables=[self.current_ct.__table__,
                                                 self.current_oh.__table__],
                                         checkfirst=True)

                solution = self.manage_any_optimization()

                optimization_task = self.query_current_optimization_task()

                optimization_task.solution = solution
                optimization_task.optimization_state = OPTIMIZATION_FINISH
                self.session.commit()

                # Remove single job properties
                self.remove_optimization_and_calculation_data()

                continue

            self.remove_old_optimization_tasks_and_tables()

            # print("No jobs. Sleeping for 1 minute.")
            # sleep(60)


if __name__ == '__main__':
    sleep(10)

    optimization_manager = OptimizationManager(
        session=Session,
        ea_toolbox=EAToolbox,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask,
        optimization_history=OptimizationHistory,
        debug=True
    )

    optimization_manager.run()
