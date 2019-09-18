import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from sympy import lambdify, Symbol

from helper_functions import load_json, write_json
from db import Session
from models import CalculationTask, OptimizationTask
from config import CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH
from .x_helper import g_mod

G_MOD_CONST = 10000

X_SYMB = Symbol("x")


class WorkerManager:
    def __init__(self,
                 session,
                 optimization_task,
                 calculation_task,
                 debug: bool = False):
        self.session = session
        self.optimization_task = optimization_task
        self.calculation_task = calculation_task
        self.debug = debug

    def query_first_starting_calculationtask(self) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        return self.session.query(self.calculation_task)\
            .filter(self.calculation_task.calculation_state == CALCULATION_START).first()

    def query_optimizationtask_with_id(self,
                                       optimization_id):

        return self.session.query(self.optimization_task).\
            filter(self.optimization_task.optimization_id == optimization_id)

    def run(self):
        while True:
            if self.debug:
                print('No jobs, sleeping for 1 minute')
                sleep(60)

            new_calculation_task = self.query_first_starting_calculationtask()

            if new_calculation_task:
                new_calculation_task.calculation_state = CALCULATION_RUN
                Session.commit()

                data_input = load_json(new_calculation_task.data_filepath)
                calculation_parameters = load_json(new_calculation_task.calc_input_filepath)

                x = g_mod(array=calculation_parameters["ind_genes"],
                          const=G_MOD_CONST)

                optimization_functions = data_input["functions"]

                data_output = dict()
                data_output["ind_genes"] = calculation_parameters["ind_genes"]
                data_output["functions"] = dict()

                for function, function_dict in optimization_functions.items():
                    f = lambdify(X_SYMB, function_dict["function"])
                    data_output["functions"][function] = f(x)

                write_json(obj=data_output,
                           filepath=new_calculation_task.calc_output_filepath)

                new_calculation_task.calculation_type = CALCULATION_FINISH
                optimization_task = self.query_optimizationtask_with_id(
                    optimization_id=new_calculation_task.optimization_id)
                optimization_task.current_population += 1
                Session.commit()


if __name__ == '__main__':
    worker_manager = WorkerManager(
        session=Session,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask,
        debug=True
    )

    worker_manager.run()
