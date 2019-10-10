import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from sympy import lambdify, Symbol

from helper_functions import load_json, write_json, get_table_for_optimization_id
from db import Session
from models import CalculationTask, OptimizationTask
from config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, OPTIMIZATION_TYPE_EVOLUTION
from x_helper import g_mod

G_MOD_CONST = 10000

X_SYMB = Symbol("x")


class WorkerManager:
    def __init__(self,
                 session,
                 optimization_task,
                 calculation_task):
        self.session = session
        self.ot = optimization_task
        self.ct = calculation_task
        # self.ct_lo = calculation_task_linear_optimization

    def query_first_starting_calculation_task(self,
                                              ct_table) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        return self.session.query(ct_table)\
            .filter(ct_table.calculation_state == CALCULATION_START).first()

    def query_calculation_task_with_id(self,
                                       ct_table,
                                       calculation_id) -> Session.query:

        return self.session.query(ct_table).\
            filter(ct_table.calculation_id == calculation_id).first()

    def query_optimization_task_with_id(self,
                                        optimization_id) -> Session.query:

        return self.session.query(self.ot).\
            filter(self.ot.optimization_id == optimization_id).first()

    def run(self):
        while True:
            running_optimization_task = self.session.query(self.ot)\
                .filter(self.ot.optimization_state == OPTIMIZATION_RUN).first()

            if running_optimization_task:
                optimization_id = running_optimization_task.optimization_id

                individual_ct = get_table_for_optimization_id(CalculationTask, optimization_id)

                new_calculation_task = self.query_first_starting_calculation_task(individual_ct)

                if new_calculation_task:
                    calculation_id = new_calculation_task.calculation_id

                    # calculation_task = self.query_calculation_task_with_id(ct_table=ct_table,
                    #                                                        calculation_id=calculation_id)
                    # print(f"Working on task with id: {calculation_id}")

                    new_calculation_task.calculation_state = CALCULATION_RUN
                    self.session.commit()

                    data_input = load_json(new_calculation_task.data_filepath)
                    calculation_parameters = load_json(new_calculation_task.calcinput_filepath)

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
                               filepath=new_calculation_task.calcoutput_filepath)

                    # print("Wrote json.")
                    #
                    # calculation_task = self.query_calculation_task_with_id(ct_table=ct_table,
                    #                                                        calculation_id=calculation_id)
                    new_calculation_task.calculation_state = CALCULATION_FINISH

                    if running_optimization_task.optimization_type == OPTIMIZATION_TYPE_EVOLUTION:
                        running_optimization_task.current_population += 1
                    self.session.commit()

                    # print("Session committed.")

                    continue

            # print("No jobs. Sleeping for 30 seconds.")
            # sleep(30)


if __name__ == '__main__':
    sleep(10)

    worker_manager = WorkerManager(
        session=Session,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask
    )

    worker_manager.run()
