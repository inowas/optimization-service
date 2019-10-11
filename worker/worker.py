import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from sympy import lambdify, Symbol
from pathlib import Path
import flopy
import numpy as np
from typing import Union, List

from helper_functions import load_json, write_json, get_table_for_optimization_id
from db import Session
from models import CalculationTask, OptimizationTask
from numpy_function_mapping import STRING_TO_NUMPY_FUNCTION
from config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_TYPE_EVOLUTION, CALCULATION_DATA, MODFLOW_EXE
# from x_helper import g_mod

MISSING_DATA_VALUE = -9999
HEADFILE_EXT = ".hds"


G_MOD_CONST = 10000

X_SYMB = Symbol("x")


class ModflowModel:

    def __init__(self,
                 optimization_data: dict,
                 data: dict,
                 model_name: str,
                 model_path: Union[str, Path]):
        self.optimization_data = optimization_data
        self.data = data
        self.mf = None
        self.mt = None

        self.optimization_data = optimization_data
        self.mf_parameters = data["mf"]["mf"]
        self.mf_parameters["modelname"] = model_name
        self.mf_parameters["model_ws"] = model_path

        self.mt_parameters = None
        if data.get("mt"):
            self.mt_parameters = data["mt"]["mt"]

        self.mf_packages = None
        self.mt_packages = None
        # self.added_packages = None
        #
        # for package_name in data["mf"]["packages"]:
        #     self.mf_packages[package_name] = data["mf"][package_name]

    def get_fitness(self):
        pass

    def check_constraints(self):
        for constraint in self.optimization_data["constraints"]:
            mask = None
            if constraint["type"] in ["concentration", "head"]:
                mask = self.make_mask(constraint["location"],
                                      self.optimization_data["objects"],
                                      self.mf.get_package("dis"))

            if constraint["type"] == "concentration":




    @staticmethod
    def create_default_package(package):
        # todo create a default package instantiator
        pass

    @staticmethod
    def combine_package_with_object(package, obj):
        # todo create a function ti include an object in a package (e.g. a single well in well package)
        return package

    # Author: Aybulat Fatkhutdinov
    @staticmethod
    def make_mask(location, objects, dis_package):
        """ Returns an array mask of location that has nper,nlay,nrow,ncol dimensions """

        nstp_flat = dis_package.nstp.array.sum()
        nrow = dis_package.nrow
        ncol = dis_package.ncol
        nlay = dis_package.nlay

        mask = None

        if location["type"] == 'bbox':
            try:
                per_min = location['ts']['min']
            except KeyError:
                per_min = 0

            try:
                per_max = location['ts']['max']
            except KeyError:
                per_max = nstp_flat

            try:
                lay_min = location['lay']['min']
            except KeyError:
                lay_min = 0

            try:
                lay_max = location['lay']['max']
            except KeyError:
                lay_max = nlay

            try:
                col_min = location['col']['min']
            except KeyError:
                col_min = 0

            try:
                col_max = location['col']['max']
            except KeyError:
                col_max = ncol

            try:
                row_min = location['row']['min']
            except KeyError:
                row_min = 0

            try:
                row_max = location['row']['min']
            except KeyError:
                row_max = nrow

            if per_min == per_max:
                per_max += 1
            if lay_min == lay_max:
                lay_max += 1
            if row_min == row_max:
                row_max += 1
            if col_min == col_max:
                col_max += 1

            mask = np.zeros((nstp_flat, nlay, nrow, ncol), dtype=bool)
            mask[per_min:per_max,
                 lay_min:lay_max,
                 row_min:row_max,
                 col_min:col_max] = True

        elif location["type"] == 'object':
            lays = []
            rows = []
            cols = []
            for obj in objects:
                if obj['id'] in location['objects']:
                    lays.append(obj['position']['lay']['result'])
                    rows.append(obj['position']['row']['result'])
                    cols.append(obj['position']['col']['result'])

            mask = np.zeros((nstp_flat, nlay, nrow, ncol), dtype=bool)
            mask[:, lays, rows, cols] = True

        return mask

    @staticmethod
    def read_distance(data, objects):
        """ Returns distance between two groups of objects """

        try:
            location_1 = data["location_1"]
            location_2 = data["location_2"]

            objects_1 = None
            objects_2 = None

            if location_1['type'] == 'object':
                objects_1 = [
                    obj for obj in objects if obj['id'] in location_1['objects']
                ]

            if location_2['type'] == 'object':
                objects_2 = [
                    obj for obj in objects if obj['id'] in location_2['objects']
                ]

            distances = []
            if objects_1 is not None:
                for obj_1 in objects_1:
                    if objects_2 is not None:
                        for obj_2 in objects_2:
                            dx = float(abs(obj_2['position']['col']['result'] - obj_1['position']['col']['result']))
                            dy = float(abs(obj_2['position']['row']['result'] - obj_1['position']['row']['result']))
                            dz = float(abs(obj_2['position']['lay']['result'] - obj_1['position']['lay']['result']))
                            distances.append(np.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2)))
                    else:
                        dx = float(abs(location_2['col']['min'] - obj_1['position']['col']['result']))
                        dy = float(abs(location_2['row']['min'] - obj_1['position']['row']['result']))
                        dz = float(abs(location_2['lay']['min'] - obj_1['position']['lay']['result']))
                        distances.append(np.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2)))
            else:
                if objects_2 is not None:
                    for obj_2 in objects_2:
                        dx = float(abs(obj_2['position']['col']['result'] - location_1['col']['min']))
                        dy = float(abs(obj_2['position']['row']['result'] - location_1['row']['min']))
                        dz = float(abs(obj_2['position']['lay']['result'] - location_1['lay']['min']))
                        distances.append(np.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2)))
                else:
                    dx = float(abs(location_2['col']['min'] - location_1['col']['min']))
                    dy = float(abs(location_2['row']['min'] - location_1['row']['min']))
                    dz = float(abs(location_2['lay']['min'] - location_1['lay']['min']))
                    distances.append(np.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2)))

            distances = np.array(distances)
        except:
            distances = None

        return distances

    def add_objects(self, objects):
        for obj in objects:
            obj_package_name = obj["type"]

            if obj_package_name not in self.mf_packages:
                self.mf_packages[obj_package_name] = self.create_default_package(obj_package_name)

            self.mf_packages[obj_package_name] = self.combine_package_with_object(
                package=self.mf_packages[obj_package_name], obj=obj)

    def read_objective(self, objective):
        if objective["type"] == "concentration":
            obj_file = flopy.utils.UcnFile(filename=objective["conc_file_name"])
            obj_data = obj_file.get_alldata(nodata=MISSING_DATA_VALUE)
            obj_file.close()

        elif objective["type"] == "head":
            obj_file = flopy.utils.HeadFile(
                f'{Path(self.mf_parameters["model_ws"], self.mf_parameters["modelname"])}{HEADFILE_EXT}')
            obj_data = obj_file.get_alldata(nodata=MISSING_DATA_VALUE)
            obj_file.close()

        elif objective["type"] == "distance":
            value = self.read_distance(objective, self.objects)

        elif objective["type"] == "flux":
            value = self.read_flux(objective, self.objects)


        elif objective["type"] == "input_concentration":
            value = self.read_input_concentration(objective, self.objects)

        return obj_data

    def get_objectives(self,
                       objectives: List[dict]):
        objective_extraction = []
        for objective in objectives:
            obj_data = None
            mask = None
            if objective["type"] in ["concentration", "head"]:
                mask = self.make_mask(objective["location"],
                                      self.optimization_data["objects"],
                                      self.mf.get_package("DIS"))

            if objective["type"] == "concentration":
                obj_file = flopy.utils.UcnFile(filename=objective["conc_file_name"])
                obj_data = obj_file.get_alldata(nodata=MISSING_DATA_VALUE)
                obj_file.close()

            if objective["type"] == "head":
                obj_file = flopy.utils.HeadFile(
                    f'{Path(self.mf_parameters["model_ws"], self.mf_parameters["modelname"])}{HEADFILE_EXT}')
                obj_data = obj_file.get_alldata(nodata=MISSING_DATA_VALUE)
                obj_file.close()

            data_in_bbox = obj_data[mask]

            current_objective_value = STRING_TO_NUMPY_FUNCTION[objective["summary_method"]](data_in_bbox)[0]

            objective_extraction.append(current_objective_value)

        return objective_extraction

    def initiate_model(self):
        self.mf = flopy.modflow.Modflow(**self.mf_parameters)

        mf_packages = []
        for package_name in self.data["mf"]["packages"]:
            assert package_name in self.mf.mfnam_packages, f"Error: {package_name} not available as package"
            package_class = self.mf.mfnam_packages[package_name]

            package = package_class(self.mf, **self.data["mf"][package_name])
            mf_packages.append(package)

        self.mf_packages = mf_packages

        if self.mt_parameters:
            self.mt = flopy.mt3d.mt.Mt3dms(**self.mt_parameters)

            mt_packages = []
            for package_name in self.data["mt"]["packages"]:
                package_class = self.mt.mtnam_packages[package_name]

                package = package_class(self.mt, **self.data["mt"][package_name])

                mt_packages.append(package)

            self.mt_packages = mt_packages

    def run(self):
        self.mf.run_model()
        self.mf.write_input()
        if self.mt:
            self.mt.run_model()
            self.mt.write_input()


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
                    optimization_data = load_json(new_calculation_task.optimization_filepath)
                    calculation_data = load_json(new_calculation_task.calcinput_filepath)

                    # Build model
                    model_name = calculation_id
                    model_path = Path(CALCULATION_DATA, calculation_id)

                    modflowmodel = ModflowModel(optimization_data=optimization_data,
                                                mf_data=data["mf"],
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
