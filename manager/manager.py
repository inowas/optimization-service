import os.path
from pathlib import Path
from copy import deepcopy
from uuid import uuid4
from time import sleep
from typing import Tuple, List, Optional, Union
from sqlalchemy import and_
from sqlalchemy.exc import DBAPIError
from flopyAdapter import ModflowDataModel, FlopyFitnessAdapter

from evolutionary_toolbox import EAToolbox

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "opt_app"))
from helpers.functions import load_json, write_json, get_table_for_optimization_id, get_schema_and_refresolver   # noqa: E402
from db import Session, engine  # noqa: E402
from models import Base, OptimizationTask, CalculationTask, OptimizationHistory  # noqa: E402
from helpers.config import OPTIMIZATION_DATA, OPTIMIZATION_FOLDER, CALCULATION_FOLDER, INDIVIDUAL_PARAMETERS_FOLDER, \
    ODATA_FILENAME, MDATA_FILENAME, JSON_ENDING, \
    SCHEMA_MODFLOW_MODEL_DATA, STATUS_REGULAR_CALCULATION  # noqa: E402
from helpers.config import OPTIMIZATION_START, CALCULATION_START, OPTIMIZATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_FINISH, OPTIMIZATION_ABORT


class OptimizationManager:
    def __init__(self,
                 session: Session,
                 ea_toolbox: EAToolbox,
                 optimization_task,
                 calculation_task,
                 optimization_history):
        # Db session and evolutionary algorithm toolbox
        self._session = session
        self._ea_toolbox = ea_toolbox

        # Preset tables
        self._ot_model = optimization_task
        self._oh_model_template = optimization_history
        self._oh_model = None
        # self._ct_model_template = calculation_task
        self._ct_model = calculation_task
        # self._ct_model = calculation_task

        # Json schema for calculation data
        self._schema = None
        self._refresolver = None

        # Temporary attributes (used for one optimization and then overwritten)
        self._current_oid = None

        self._current_rdata = None  # request data
        self._current_odata = None  # optimization data
        self._current_mdata = None  # model data

        self._current_vmap = None  # variable map

        self._current_eat = None
        # self._current_oh = None
        # self._current_ct = None

    def reset_temporary_attributes(self):
        self._current_oid = None

        self._current_rdata = None
        self._current_odata = None
        self._current_mdata = None
        self._current_vmap = None

        self._current_eat = None
        # self._current_oh = None
        # self._current_ct = None

    # Author: Aybulat Fatkhutdinov; modified
    @staticmethod
    def apply_individual(optimization_data, individual, variable_map):
        """Write individual values to variable template and return the filled template"""
        optimization_data = deepcopy(optimization_data)
        for ind_value, keys in zip(individual, variable_map):
            if keys[1] == 'position':
                ind_value = int(ind_value)

            if keys[1] == 'concentration':
                for object_ in optimization_data["objects"]:
                    if object_['id'] == keys[0]:
                        object_[keys[1]][keys[2]][keys[3]]['result'] = ind_value
                        break
            else:
                for object_ in optimization_data["objects"]:
                    if object_['id'] == keys[0]:
                        object_[keys[1]][keys[2]]['result'] = ind_value
                        break

        return optimization_data

    @staticmethod
    def create_unique_id():
        """ Function to generate a unique id for each calculation task

        :return:
        """
        return str(uuid4())

    # Author: Aybulat Fatkhutdinov; modified
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

        optimization_data = optimization_data or self._current_odata

        for object_ in optimization_data["objects"]:
            for parameter, value in object_.items():
                if parameter in ["position", "flux", "concentration"]:
                    for axis_or_period, data in value.items():
                        try:
                            if data['min'] != data['max']:
                                var_map.append((object_['id'], 'position', axis_or_period))
                                var_bounds.append((data['min'], data['max']))
                                initial_values.append(
                                    data.get('result', (data['max'] + data['min']) // 2))

                                object_[parameter][axis_or_period]['result'] = None
                            else:
                                object_[parameter][axis_or_period]['result'] = data['min']

                        except KeyError:
                            for component, component_data in data.items():
                                if component_data['min'] != component_data['max']:
                                    var_map.append((object_['id'], 'concentration', axis_or_period, component))
                                    var_bounds.append((component_data['min'], component_data['max']))
                                    initial_values.append(
                                        component_data.get('result', (data['max'] + data['min']) // 2))

                                    object_[parameter][axis_or_period][component]['result'] = None
                                else:
                                    object_[parameter][axis_or_period][component]['result'] = component_data['min']

        # for object_ in optimization_data["objects"]:
        #     for parameter, value in object_.items():
        #         if parameter == 'position':
        #             for axis, axis_data in value.items():
        #                 if axis_data['min'] != axis_data['max']:
        #                     var_map.append((object_['id'], 'position', axis))
        #                     var_bounds.append((axis_data['min'], axis_data['max']))
        #                     initial_value = axis_data.get('result')
        #                     if initial_value is None:
        #                         initial_values.append(int((axis_data['max'] + axis_data['min']) / 2))
        #                     else:
        #                         initial_values.append(initial_value)
        #                     object_['position'][axis]['result'] = None
        #                 else:
        #                     object_['position'][axis]['result'] = axis_data['min']
        #
        #         elif parameter == 'flux':
        #             for period, period_data in value.items():
        #                 if period_data['min'] != period_data['max']:
        #                     var_map.append((object_['id'], 'flux', period))
        #                     var_bounds.append((period_data['min'], period_data['max']))
        #                     initial_value = period_data.get('result')
        #                     if initial_value is None:
        #                         initial_values.append((period_data['max'] + period_data['min']) / 2)
        #                     else:
        #                         initial_values.append(initial_value)
        #                     object_['flux'][period]['result'] = None
        #                 else:
        #                     object_['flux'][period]['result'] = period_data['min']
        #
        #         elif parameter == 'concentration':
        #             for period, period_data in value.items():
        #                 for component, component_data in period_data.items():
        #                     if component_data['min'] != component_data['max']:
        #                         var_map.append((object_['id'], 'concentration', period, component))
        #                         var_bounds.append((component_data['min'], component_data['max']))
        #                         initial_value = component_data.get('result')
        #                         if initial_value is None:
        #                             initial_values.append((component_data['max'] + component_data['min']) / 2)
        #                         else:
        #                             initial_values.append(initial_value)
        #                         object_[parameter][period][component]['result'] = None
        #                     else:
        #                         object_[parameter][period][component]['result'] = component_data['min']

        return var_map, var_bounds, initial_values

    @property
    def optimization_weights(self) -> tuple:
        """ Function used to extract objectives from the json that holds the whole optimization task

        Args:
            self (self) - optimization_data from optimization json

        Returns:
             objectives (tuple of ints) - the objectives of each function as presented in the json
        """
        return tuple(objective["weight"] for objective in self._current_odata["objectives"])

    def linear_scalarization(self,
                             fitness: List[float]):
        scalar_fitness = 0
        for value, weight in zip(fitness, self.optimization_weights):
            scalar_fitness += value * weight * -1
        return scalar_fitness

    @property
    def first_starting_ot(self) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        return self._session.query(self._ot_model)\
            .filter(self._ot_model.optimization_state == OPTIMIZATION_START).first()

    @property
    def current_ot(self) -> Session.query:
        """

        :param self:
        :return:
        """

        return self._session.query(self._ot_model).\
            filter(self._ot_model.optimization_id == self._current_oid).first()

    def latest_scalar_fitness_of_linear_optimization(self):
        return self._session.query(self._ot_model).\
            filter(self._ot_model.optimization_id == self.current_ot.optimization_id).\
            order_by(self._ot_model.optimization_id.desc()).first().fitness

    def query_finished_calculation_tasks(self,
                                         generation: int) -> Session.query:
        """

        :param generation:
        :return:
        """

        return self._session.query(self._ct_model).\
            filter(self._ct_model.generation == generation,
                   self._ct_model.calculation_state == CALCULATION_FINISH).all()

    def summarize_finished_calculation_tasks(self,
                                             generation: int) -> Tuple[List[dict], list]:
        """

        :param generation:
        :return:
        """

        finished_calculation_tasks = self.query_finished_calculation_tasks(generation=generation)

        summarized_individual_odata = []
        summarized_fitness = []

        for calculation in finished_calculation_tasks:
            # Only append if calculation was successful!
            if calculation.status == STATUS_REGULAR_CALCULATION:
                individual_odata = (Path(OPTIMIZATION_DATA) / OPTIMIZATION_FOLDER / self.current_ot.optimization_id /
                                    INDIVIDUAL_PARAMETERS_FOLDER / calculation.data_hash /
                                    f"{ODATA_FILENAME}{JSON_ENDING}")

                summarized_individual_odata.append(load_json(individual_odata))

                flopyfitnessadapter = FlopyFitnessAdapter.from_id(self._current_odata,
                                                                  calculation.data_hash,
                                                                  Path(OPTIMIZATION_DATA) / CALCULATION_FOLDER)

                summarized_fitness.append(flopyfitnessadapter.get_fitness())

        return summarized_individual_odata, summarized_fitness

    def await_generation_finished(self,
                                  generation: int,
                                  total_population: Optional[int] = None):
        """

        :param generation:
        :param total_population:
        :return:
        """
        total_population = total_population or self.current_ot.total_population

        while True:
            current_population = self._session.query(self._ct_model).\
                filter(and_(self._ct_model.optimization_id == self._current_oid,
                            self._ct_model.generation == generation,
                            self._ct_model.calculation_state == CALCULATION_FINISH)).count()

            if current_population == total_population:
                break

    def create_single_calculation_job(self,
                                      individual: List[float],
                                      generation: int = None) -> None:
        """

        :param individual:
        :param generation:
        :return:
        """

        individual_odata = self.apply_individual(optimization_data=deepcopy(self._current_odata),
                                                 individual=individual,
                                                 variable_map=self._current_vmap)

        modflowdatamodel = ModflowDataModel.from_data(data=self._current_mdata,
                                                      schema=self._schema,
                                                      resolver=self._refresolver)

        modflowdatamodel.add_objects(objects=individual_odata["objects"])

        parameter_set_filepath = (Path(OPTIMIZATION_DATA) / OPTIMIZATION_FOLDER / self.current_ot.optimization_id /
                                  INDIVIDUAL_PARAMETERS_FOLDER / modflowdatamodel.md5_hash /
                                  f"{ODATA_FILENAME}{JSON_ENDING}")

        calculation_data_filepath = (Path(OPTIMIZATION_DATA) / CALCULATION_FOLDER / modflowdatamodel.md5_hash /
                                     f"{MDATA_FILENAME}{JSON_ENDING}")

        try:
            parameter_set_filepath.parent.mkdir(parents=True)
        except FileExistsError:
            pass

        try:
            calculation_data_filepath.parent.mkdir(parents=True)
        except FileExistsError:
            pass

        try:
            write_json(obj=individual_odata,
                       filepath=parameter_set_filepath)

            existing_jobs_with_cid = self._session.query(self._ct_model)\
                .filter(self._ct_model.calculation_id == modflowdatamodel.md5_hash).all()

            if not existing_jobs_with_cid:
                write_json(obj=modflowdatamodel.data,
                           filepath=calculation_data_filepath)

            new_calc_task = self._ct_model(
                optimization_id=self.current_ot.optimization_id,
                calculation_id=self.create_unique_id(),
                data_hash=modflowdatamodel.md5_hash,
                calculation_type=self.current_ot.optimization_type,
                calculation_state=CALCULATION_START,  # Set state to start
                generation=generation
            )

            self._session.add(new_calc_task)
            self._session.commit()

        except (IOError, DBAPIError) as e:
            print(str(e))
            self._session.rollback()

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
                                               individual=individual)

    def linear_optimization_queue(self,
                                  individual):  # : List[float]
        """

        :param individual:
        :return:
        """

        calculation_task = self._session.query(self._ct_model).\
            order_by(self._ct_model.optimization_id.desc(), self._ct_model.calculation_id.desc()).first()

        try:
            generation = calculation_task.generation + 1
        except TypeError:  # when generation is None, fresh start
            generation = 1

        total_population = 1

        individual = list(individual)  # Mystic solver converts solution to a numpy array; we need a list

        self.create_single_calculation_job(generation=generation,
                                           individual=individual)

        self.await_generation_finished(generation=generation,
                                       total_population=total_population)

        _, summarized_fitness = self.summarize_finished_calculation_tasks(generation=generation)

        scalar_solution = self.linear_scalarization(summarized_fitness[0]["fitness"])

        optimization_task = self.current_ot

        optimization_task.fitness = [scalar_solution]
        self._session.commit()

        return scalar_solution

    def manage_evolutionary_optimization(self) -> List[List[float]]:
        """ Function for evolutionary optimization
        The function first grabs the parameters number of generations and population size and then starts
        walking through the generations and each time
        - getting last generation fitnesses with parameters and developing a new population based on the fitness
        of each(except for first generation)
        - creating new calculation jobs for those new individuals
        - waiting for finished calculations of generation
        - again getting parameters and fitnesses, this time of current generation for creating a new population
        based on the fitness
        - selecting the best of the evaluated population

        :return:
        """
        number_of_generations = self._current_odata["parameters"]["ngen"]
        population_size = self._current_odata["parameters"]["pop_size"]

        population = self._current_eat.make_population(population_size)

        for generation in range((number_of_generations+1)):

            optimization_task = self.current_ot

            if (optimization_task.optimization_state == OPTIMIZATION_ABORT and
                    optimization_task.current_generation > 1):
                break

            optimization_task.current_generation = generation + 1  # Table counts to ten
            optimization_task.current_population = 0
            self._session.commit()

            # todo modify the following code to reduce code repetition
            if generation > 0:
                summarized_individual_odata, summarized_fitness = self.summarize_finished_calculation_tasks(
                    generation=(generation - 1)
                )

                individuals = [
                    self.read_optimization_data(optimization_data)[2]
                    for optimization_data in summarized_individual_odata
                ]

                fitnesses = [self.linear_scalarization(single_fitness)
                             for single_fitness in summarized_fitness]

                population = self._current_eat.optimize_evolutionary(individuals, fitnesses)

            self.create_new_calculation_jobs(generation, population)

            self.await_generation_finished(generation)

            summarized_individual_odata, summarized_fitness = self.summarize_finished_calculation_tasks(
                generation)

            individuals = [self.read_optimization_data(optimization_data)[2]
                           for optimization_data in summarized_individual_odata]

            fitnesses = [self.linear_scalarization(single_fitness)
                         for single_fitness in summarized_fitness]

            population = self._current_eat.evaluate_finished_calculations(individuals, fitnesses)

            population = self._current_eat.select_best_individuals(population)

            optimization_history = self._oh_model(
                author=optimization_task.author,
                project=optimization_task.project,
                optimization_id=self._current_oid,
                generation=(generation + 1),
                scalar_fitness=self.linear_scalarization(
                    self._current_eat.select_nth_of_hall_of_fame(1)[0].fitness.values)
            )

            self._session.add(optimization_history)
            self._session.commit()

        return self._current_eat.get_solutions_and_fitnesses(
            self._current_odata["parameters"]["pop_size"])

    def manage_linear_optimization(self) -> List[float]:
        """ Manager for linear optimizations. It only calls the linear optimization function of the ea toolbox and
        passes it the solution as given by the optimization data along with the linear optimization queue function
        which manages to distribute a calculation tasks and also summarizes the solution. This has to happen all in one,
        as the linear optimization function NSGA2 doesn't have any kind of generations and just continues until a
        certain threshold is undergone.

        :return:
        """

        self._current_eat.optimize_linear(initial_values=self.read_optimization_data()[2],
                                          function=self.linear_optimization_queue,
                                          fitness_retriever=self.latest_scalar_fitness_of_linear_optimization)

        return self._current_eat.get_solutions_and_fitnesses()

    def manage_any_optimization(self) -> Union[List[float], List[List[float]]]:
        """ Manager for any kind of optimization that handles both offered types (genetic and linear). Primarily this
        method was introduced to separate the two methods and have only one function call that divides depending on
        the current optimization task.

        :return:
        """
        optimization_task = self._session.query(self._ot_model)\
            .filter(self._ot_model.optimization_id == self._current_oid).first()

        assert optimization_task.optimization_type in ["GA", "Simplex"], \
            "Error: optimization_type is neither 'GA' nor 'Simplex'"

        if optimization_task.optimization_type == "GA":
            return self.manage_evolutionary_optimization()

        if optimization_task.optimization_type == "Simplex":
            return self.manage_linear_optimization()

    # def remove_optimization_and_calculation_data(self) -> None:
    #     optimization_task = self.current_ot
    #
    #     Path(optimization_task.data_filepath).unlink()
    #
    #     individual_ct = get_table_for_optimization_id(self._ct_model_template, self._current_oid)
    #
    #     calculations = self._session.query(individual_ct).all()
    #
    #     calculation_files = [(calculation.calcinput_filepath,  calculation.calcoutput_filepath)
    #                          for calculation in calculations]
    #
    #     for calcinput_file, calcoutput_file in calculation_files:
    #         Path(calcinput_file).unlink()
    #         Path(calcoutput_file).unlink()

    # def remove_old_optimization_tasks_and_tables(self) -> None:
    #     now_date = datetime.now().date()
    #
    #     optimization_tasks = self._session.query(self._ot_model)\
    #         .filter(or_(self._ot_model.optimization_state == OPTIMIZATION_FINISH,
    #                     self._ot_model.optimization_state == OPTIMIZATION_ABORT)).all()
    #
    #     old_optimization_tasks = []
    #     for task in optimization_tasks:
    #         existing_time = (now_date - pd.to_datetime(task.publishing_date).date()).days
    #         if existing_time > MAX_STORING_TIME_OPTIMIZATION_TASKS:
    #             old_optimization_tasks.append(task)
    #
    #     for task in old_optimization_tasks:
    #         # individual_ct = get_table_for_optimization_id(self._ct_model_template, self._current_oid)
    #         individual_oh = get_table_for_optimization_id(self._oh_model_template, self._current_oid)
    #
    #         Base.metadata.drop_all(tables=[individual_oh], bind=engine)
    #
    #         self._session.remove(task)
    #         self._session.commit()

    def run(self):
        """ Function run is used to keep the manager working constantly. It will work on one optimization only and
        fulfill the job which includes constantly creating jobs for one generation, then after calculation
        summarizing the results and creating new generations with new jobs and finally put the solution back in the
        table and set the optimization to be finished.

        Returns:
            None - output is managed over databases

        """

        while True:

            if self.first_starting_ot:
                print(f"Working on task with id: {self.first_starting_ot.optimization_id}")

                self._current_oid = self.first_starting_ot.optimization_id

                optimization_task = self.current_ot
                optimization_task.optimization_state = OPTIMIZATION_RUN
                self._session.commit()

                # Set temporary attributes
                request_data = load_json(
                    Path(OPTIMIZATION_DATA) / OPTIMIZATION_FOLDER /
                    optimization_task.optimization_id / "optimization.json")

                self._current_rdata = {
                    request_data[key]
                    for key in request_data
                    if key not in ["optimization", "data"]
                }
                self._current_odata = request_data["optimization"]
                self._current_mdata = request_data["data"]

                # Set temporary schema, solver
                self._schema, self._refresolver = get_schema_and_refresolver(SCHEMA_MODFLOW_MODEL_DATA)

                variable_map, variable_bounds, _ = self.read_optimization_data()

                self._current_vmap = variable_map

                # Set temporary attributes
                self._current_eat = self._ea_toolbox.from_data(
                    bounds=variable_bounds,
                    weights=self.optimization_weights,  # from self.optimization_data
                    parameters=self._current_odata["parameters"]
                )
                self._oh_model = get_table_for_optimization_id(self._oh_model_template, self._current_oid)
                # self._ct_model = get_table_for_optimization_id(self._ct_model_template, self._current_oid)

                Base.metadata.create_all(bind=engine,
                                         tables=[self._oh_model.__table__],
                                         checkfirst=True)

                solutions, fitnesses = self.manage_any_optimization()

                optimization_task = self.current_ot

                optimization_task.solution = solutions
                optimization_task.fitness = [self.linear_scalarization(fitness) for fitness in fitnesses]
                optimization_task.optimization_state = OPTIMIZATION_FINISH
                self._session.commit()

                # Remove single job properties
                # self.remove_optimization_and_calculation_data()

                continue

            # self.remove_old_optimization_tasks_and_tables()

            # print("No jobs. Sleeping for 1 minute.")
            # sleep(60)


if __name__ == '__main__':
    sleep(10)

    optimization_manager = OptimizationManager(
        session=Session,
        ea_toolbox=EAToolbox,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask,
        optimization_history=OptimizationHistory
    )

    optimization_manager.run()
