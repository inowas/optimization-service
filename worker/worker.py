import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from sympy import lambdify, Symbol

from helper_functions import load_json, write_json
from db import Session
from models import CalculationTaskEvolutionaryOptimization, CalculationTaskLinearOptimization, OptimizationTask
from config import CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH
from x_helper import g_mod

G_MOD_CONST = 10000

X_SYMB = Symbol("x")


class WorkerManager:
    def __init__(self,
                 session,
                 optimization_task,
                 calculation_task_evolutionary_optimization,
                 calculation_task_linear_optimization):
        self.session = session
        self.ot = optimization_task
        self.ct_eo = calculation_task_evolutionary_optimization
        self.ct_lo = calculation_task_linear_optimization

    def query_first_starting_calculation_task(self) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        calculation_task = self.session.query(self.ct_eo)\
            .filter(self.ct_eo.calculation_state == CALCULATION_START).first()

        if not calculation_task:
            return self.ct_lo, self.session.query(self.ct_lo) \
                .filter(self.ct_lo.calculation_state == CALCULATION_START).first()

        return self.ct_eo, calculation_task

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
            ct_table, new_calculation_task = self.query_first_starting_calculation_task()

            if new_calculation_task:
                calculation_id = new_calculation_task.calculation_id

                calculation_task = self.query_calculation_task_with_id(ct_table=ct_table,
                                                                       calculation_id=calculation_id)
                print(f"Working on task with id: {calculation_id}")
                calculation_task.calculation_state = CALCULATION_RUN
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

                print("Wrote json.")

                calculation_task = self.query_calculation_task_with_id(ct_table=ct_table,
                                                                       calculation_id=calculation_id)
                calculation_task.calculation_state = CALCULATION_FINISH
                optimization_task = self.query_optimization_task_with_id(
                    optimization_id=calculation_task.optimization_id)
                optimization_task.current_population += 1
                self.session.commit()

                print("Session committed.")


if __name__ == '__main__':
    worker_manager = WorkerManager(
        session=Session,
        optimization_task=OptimizationTask,
        calculation_task_evolutionary_optimization=CalculationTaskEvolutionaryOptimization,
        calculation_task_linear_optimization=CalculationTaskLinearOptimization
    )

    worker_manager.run()
