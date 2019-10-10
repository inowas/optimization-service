import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from sympy import lambdify, Symbol
from pathlib import Path
import flopy
from typing import Union

from helper_functions import load_json, write_json, get_table_for_optimization_id
from db import Session
from models import CalculationTask, OptimizationTask
from config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_TYPE_EVOLUTION, CALCULATION_DATA, MODFLOW_EXE
# from x_helper import g_mod


G_MOD_CONST = 10000

X_SYMB = Symbol("x")


class ModflowModel:

    def __init__(self, mf_data: dict,
                 model_name: str,
                 model_path: Union[str, Path]):
        self.mf = None

        self.mf_parameters = mf_data["mf"]
        self.mf_parameters["modelname"] = model_name
        self.mf_parameters["model_ws"] = model_path

        self.mf_packages = dict()
        self.added_packages = None

        for package_name in mf_data["packages"]:
            self.mf_packages[package_name] = mf_data[package_name]

    @staticmethod
    def create_default_package(package):
        # todo create a default package instantiator
        pass

    @staticmethod
    def combine_package_with_object(package, object):
        # todo create a function ti include an object in a package (e.g. a ingle well in well package)
        return package

    def add_objects(self, objects):
        for obj in objects:
            obj_package_name = obj["type"]

            if obj_package_name not in self.mf_packages:
                self.mf_packages[obj_package_name] = self.create_default_package(obj_package_name)

            self.mf_packages[obj_package_name] = self.combine_package_with_object(
                package=self.mf_packages[obj_package_name], object=obj)

    def get_objectives(self,
                       objectives):
        objective_extraction = []
        for objective in objectives:
            objetive_name = objective["type"]

            if objetive_name == "concentration":
                ucnobj = flopy.utils.UcnFile(filename=objective["conc_file_name"])
                ucndata = ucnobj.get_alldata()

                ntimes, nlay, nrow, ncol = [objective["location"][key] for key in ["ts", "lay", "row", "col"]]

                ucndata[ntimes[0]:(ntimes[1] + 1), nlay[0]:(nlay[1] + 1), nrow[0]:(nrow[1] + 1), ncol[0]:(ncol[1] + 1)]


    def initiate_model(self):
        mf = flopy.modflow.Modflow(**self.mf_parameters)

        for package_name, package_parameters in self.mf_packages.items():
            assert package_name in mf.mfnam_packages, f"Error: {package_name} not available as package"
            package_class = mf.mfnam_packages[package_name]

            package = package_class(mf, **package_parameters)
            self.added_packages.append(package)

        self.mf = mf

    def run(self):
        self.mf.run_model()
        self.mf.write_input()



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

                    data = load_json(new_calculation_task.data_filepath)
                    # optimization = load_json(new_calculation_task.optimization_filepath)
                    calculation_data = load_json(new_calculation_task.calcinput_filepath)

                    # Build model
                    model_name = calculation_id
                    model_path = Path(CALCULATION_DATA, calculation_id)

                    modflowmodel = ModflowModel(mf_data=data["mf"],
                                                model_name=model_name,
                                                model_path=model_path)

                    modflowmodel.initiate_model()

                    modflowmodel.add_objects(objects=calculation_data["objects"])

                    modflowmodel.run()


                    # x = g_mod(array=calculation_parameters["ind_genes"],
                    #           const=G_MOD_CONST)
                    #
                    # optimization_functions = data_input["functions"]
                    #
                    # data_output = dict()
                    # data_output["ind_genes"] = calculation_parameters["ind_genes"]
                    # data_output["functions"] = dict()
                    #
                    # for function, function_dict in optimization_functions.items():
                    #     f = lambdify(X_SYMB, function_dict["function"])
                    #     data_output["functions"][function] = f(x)

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
