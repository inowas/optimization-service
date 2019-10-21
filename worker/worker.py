import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
from pathlib import Path
# import flopy
import numpy as np
from typing import Union, List

from mapping.flopy_adapter_mapping import FLOPY_PACKAGENAME_TO_ADAPTER
from mapping.numpy_function_mapping import STRING_TO_NUMPY_FUNCTION
from mapping.object_to_package_mapping import OBJECT_TO_PACKAGE_MAPPER
from InowasFlopyAdapter.InowasFlopyReadFitness import InowasFlopyReadFitness
from InowasFlopyAdapter.InowasFlopyCalculationAdapter import InowasFlopyCalculationAdapter
from helper_functions import load_json, write_json, get_table_for_optimization_id
from db import Session
from models import CalculationTask, OptimizationTask
from config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_TYPE_EVOLUTION, OPTIMIZATION_DATA

MISSING_DATA_VALUE = -9999


class ModflowModel:

    def __init__(self,
                 version: str,
                 optimization_id: str,
                 calculation_data: dict,
                 modelname: str = None,
                 modelpath: Union[str, Path] = None):
        if not isinstance(version, str):
            raise TypeError("Error: 'version' is not of type string.")
        if not isinstance(optimization_id, str):
            raise TypeError("Error: 'optimization_id' is not of type string.")
        if not isinstance(calculation_data, dict):
            raise TypeError("Error: 'data' is not of type dict.")
        if not isinstance(modelname, str):
            raise TypeError("Error: 'modelname' is not of type string.")
        if not isinstance(modelpath, (str, Path)):
            raise TypeError("Error: 'modelpath' is not of type string/Path.")

        self.version = version
        self.optimization_id = optimization_id
        self.calculation_data = calculation_data

        if "mf" in self.calculation_data["data"]:
            model_type = "mf"
        elif "mt" in self.calculation_data["data"]:
            model_type = "mt"
        else:
            model_type = None

        if modelname:
            self.calculation_data["data"][model_type][model_type]["modelname"] = modelname

        if modelpath:
            self.calculation_data["data"][model_type][model_type]["model_ws"] = modelpath

        self._model = None
        self._success = None

        self.combine_data_with_optimization_data()

    def combine_data_with_optimization_data(self):
        # Set modflow data path
        data_mf = self.calculation_data["data"]["mf"]

        for obj in self.calculation_data["optimization"]["objects"]:
            obj_type = obj["type"]
            obj_position = obj["position"]

            if obj_type not in data_mf:
                obj_adapter = FLOPY_PACKAGENAME_TO_ADAPTER[obj_type.capitalize()]
                data_mf[obj_type] = obj_adapter.default()

            # todo include other objects and use specific packages for the case of more parameters

            if obj_type == "well":
                package_type = OBJECT_TO_PACKAGE_MAPPER["well"]

                if not data_mf[obj_type]["stress_period_data"]:
                    data_mf[obj_type]["stress_period_data"] = {}

                for period, obj_flux in obj["flux"].items():
                    period_flux = [obj_position["lay"]["result"],
                                   obj_position["row"]["result"],
                                   obj_position["col"]["result"],
                                   obj_flux["result"]]

                    if period not in data_mf[package_type]["stress_period_data"]:
                        data_mf[package_type]["stress_period_data"][period] = []

                    same_position_flux = [existing_flux
                                          for existing_flux in data_mf[package_type]["stress_period_data"][period]
                                          if existing_flux[:3] == period_flux[:3]]

                    if same_position_flux:
                        same_position_flux[0][3] += period_flux[3]
                        continue

                    data_mf[package_type]["stress_period_data"][period].append(period_flux)

    def run(self):
        flopy_calculation = InowasFlopyCalculationAdapter(self.version,
                                                          self.calculation_data["data"],
                                                          self.optimization_id)

        self._model, self._success = flopy_calculation.get_model_and_fitness()

    def evaluate(self):
        if not self._success:
            return 999
        flopy_evaluation = InowasFlopyReadFitness(self.calculation_data["optimization"], self._model)

        return flopy_evaluation.get_fitness()


class WorkerManager:
    def __init__(self,
                 session,
                 optimization_task,
                 calculation_task):
        self.session = session
        self.ot = optimization_task
        self.ct = calculation_task

        # Temporary attributes
        self.current_optimization_id = None
        self.current_calculation_id = None
        self.current_ct = None

    def reset_temporary_attributes(self):
        self.current_optimization_id = None
        self.current_calculation_id = None
        self.current_ct = None

    def query_first_starting_calculation_task(self) -> Session.query:
        """

        Returns:
            query - first optimization task in list

        """
        return self.session.query(self.current_ct)\
            .filter(self.current_ct.calculation_state == CALCULATION_START).first()

    def query_calculation_task_with_id(self) -> Session.query:

        return self.session.query(self.current_ct).\
            filter(self.current_ct.calculation_id == self.current_calculation_id).first()

    def query_current_optimization_task(self) -> Session.query:

        return self.session.query(self.ot).\
            filter(self.ot.optimization_id == self.current_optimization_id).first()

    def run(self):
        while True:
            running_optimization_task = self.session.query(self.ot)\
                .filter(self.ot.optimization_state == OPTIMIZATION_RUN).first()

            if running_optimization_task:

                self.current_optimization_id = running_optimization_task.optimization_id
                self.current_ct = get_table_for_optimization_id(self.ct, self.current_optimization_id)

                new_calculation_task = self.query_first_starting_calculation_task()

                if new_calculation_task:
                    self.current_calculation_id = new_calculation_task.calculation_id

                    # calculation_task = self.query_calculation_task_with_id(ct_table=ct_table,
                    #                                                        calculation_id=calculation_id)
                    # print(f"Working on task with id: {calculation_id}")

                    new_calculation_task.calculation_state = CALCULATION_RUN
                    self.session.commit()

                    calculation_data = load_json(new_calculation_task.calcinput_filepath)

                    # Build model
                    modflowmodel = ModflowModel(version=calculation_data.get("version"),
                                                optimization_id=self.current_optimization_id,
                                                calculation_data=calculation_data,
                                                modelname=self.current_calculation_id,
                                                modelpath=Path(OPTIMIZATION_DATA,
                                                               self.current_optimization_id,
                                                               self.current_calculation_id))

                    modflowmodel.run()

                    fitness = modflowmodel.evaluate()

                    data_output = {"fitness": fitness}

                    write_json(obj=data_output,
                               filepath=new_calculation_task.calcoutput_filepath)

                    # print("Wrote json.")

                    new_calculation_task.calculation_state = CALCULATION_FINISH

                    if running_optimization_task.optimization_type == OPTIMIZATION_TYPE_EVOLUTION:
                        running_optimization_task.current_population += 1
                    self.session.commit()

                    # print("Session committed.")

                    continue


if __name__ == '__main__':
    sleep(10)

    worker_manager = WorkerManager(
        session=Session,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask
    )

    worker_manager.run()
