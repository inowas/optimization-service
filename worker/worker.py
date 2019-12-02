import os.path
from pathlib import Path
from time import sleep
from sqlalchemy import and_, or_
from flopyAdapter import ModflowDataModel, FlopyModelManager
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "opt_app"))
from db import Session, engine  # noqa: E402
from models import CalculationTask, OptimizationTask  # noqa: E402

from helpers.functions import load_json  # noqa: E402
from helpers.config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_TYPE_EVOLUTION, MDATA_FILENAME, OPTIMIZATION_DATA, CALCULATION_FOLDER, \
    JSON_ENDING, OPTIMIZATION_ABORT, STATUS_ERROR_CALCULATION, STATUS_REGULAR_CALCULATION  # noqa: E402


class WorkerManager:
    def __init__(self,
                 session,
                 optimization_task,
                 calculation_task):
        self._session = session
        self._ot_model = optimization_task
        # Table template
        # self._ct_model_template = calculation_task
        self._ct_model = calculation_task

        # Temporary attributes
        self._current_oid = None
        self._current_data_hash = None
        # self._current_ct = None

    def reset_temporary_attributes(self):
        self._current_oid = None
        self._current_data_hash = None
        # self._current_ct = None

    def await_calculation_table(self):
        """ Function for waiting for calculation table to be created before working with it

        :return:
        """
        while True:
            if engine.has_table(table_name=self._ct_model.__tablename__):
                break

    @property
    def first_starting_ct(self) -> Session.query:
        """ Return first calculation task that has not yet been started

        Returns:
            query - first optimization task in list

        """
        return self._session.query(self._ct_model)\
            .filter(self._ct_model.calculation_state == CALCULATION_START).first()

    @property
    def current_ct(self) -> Session.query:

        return self._session.query(self._ct_model).\
            filter(self._ct_model.data_hash == self._current_data_hash).first()

    @property
    def current_ot(self) -> Session.query:

        return self._session.query(self._ot_model).\
            filter(or_(self._ot_model.optimization_state == OPTIMIZATION_RUN,
                       self._ot_model.optimization_state == OPTIMIZATION_ABORT)).first()

    @property
    def any_finished_ct_with_same_id(self):

        return self._session.query(self._ct_model). \
            filter(and_(self._ct_model.data_hash == self._current_data_hash,
                        self._ct_model.calculation_state == CALCULATION_FINISH)).first()

    def run(self):
        while True:

            if self.current_ot:
                # Could happen that state is switched to finish in between and thus no more optimization is activated
                # Workaround
                try:
                    self._current_oid = self.current_ot.optimization_id
                except AttributeError:
                    continue

                if self.first_starting_ct:
                    self._current_data_hash = self.first_starting_ct.data_hash

                    existing_jobs_with_cid = self._session.query(self._ct_model)\
                        .filter(self._ct_model.data_hash == self._current_data_hash).all()

                    for job in existing_jobs_with_cid:
                        job.calculation_state = CALCULATION_RUN

                    self._session.commit()

                    if not self.any_finished_ct_with_same_id:
                        calculation_data_filepath = (Path(OPTIMIZATION_DATA) / CALCULATION_FOLDER /
                                                     self._current_data_hash / f"{MDATA_FILENAME}{JSON_ENDING}")
                        calculation_data = load_json(calculation_data_filepath)

                        # data was already validated by optimization manager
                        modflowdatamodel = ModflowDataModel(calculation_data)

                        # overwrite model_ws (".") with real folder
                        modflowdatamodel.model_ws = calculation_data_filepath.parent

                        flopymodelmanager = FlopyModelManager.from_modflowdatamodel(modflowdatamodel)

                        flopymodelmanager.build_flopymodel()

                        flopymodelmanager.run_model()

                        # status is defined by newly run model
                        if not flopymodelmanager.overall_model_success:
                            current_job_calculation_status = STATUS_ERROR_CALCULATION
                        else:
                            current_job_calculation_status = STATUS_REGULAR_CALCULATION

                    else:
                        # If already finished model, status will be taken from that
                        current_job_calculation_status = self.any_finished_ct_with_same_id.status

                    for job in existing_jobs_with_cid:
                        job.calculation_state = CALCULATION_FINISH
                        job.status = current_job_calculation_status

                        # Evolutionary optimization: tracking of population is needed for progress
                        if self.current_ct.calculation_type == OPTIMIZATION_TYPE_EVOLUTION:
                            self.current_ot.current_population += 1

                    self._session.commit()

                    continue


if __name__ == '__main__':
    sleep(10)

    worker_manager = WorkerManager(
        session=Session,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask
    )

    worker_manager.run()
