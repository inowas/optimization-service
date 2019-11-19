from pathlib import Path
from time import sleep
from sqlalchemy import and_
from flopyAdapter import ModflowDataModel, FlopyModelManager
import sys
sys.path.append(Path(__file__).resolve().parent / 'opt_app')
from db import Session  # noqa: E402
from models import CalculationTask, OptimizationTask  # noqa: E402

from app.helpers.functions import load_json, get_table_for_optimization_id  # noqa: E402
from app.helpers.config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_TYPE_EVOLUTION, MISSING_DATA_VALUE  # noqa: E402


class WorkerManager:
    def __init__(self,
                 session,
                 optimization_task,
                 calculation_task):
        self._session = session
        self._ot_model = optimization_task
        # Table template
        self._ct_model_template = calculation_task
        self._ct_model = None

        # Temporary attributes
        self._current_oid = None
        self._current_data_hash = None
        self._current_ct = None

    def reset_temporary_attributes(self):
        self._current_oid = None
        self._current_data_hash = None
        self._current_ct = None

    @property
    def first_starting_ct(self) -> Session.query:
        """ Return first calculation task that has not yet been started

        Returns:
            query - first optimization task in list

        """
        return self._session.query(self._current_ct)\
            .filter(self._current_ct.calculation_state == CALCULATION_START).first()

    @property
    def current_ct(self) -> Session.query:

        return self._session.query(self._current_ct).\
            filter(self._current_ct.data_hash == self._current_data_hash).first()

    @property
    def current_ot(self) -> Session.query:

        return self._session.query(self._ot_model).\
            filter(self._ot_model.optimization_state == OPTIMIZATION_RUN).first()

    @property
    def any_finished_ct_with_same_id(self):

        return self._session.query(self._current_ct). \
            filter(and_(self._current_ct.data_hash == self._current_data_hash,
                        self._current_ct.calculation_state == CALCULATION_FINISH)).first()

    def run(self):
        while True:

            if self.current_ot:

                self._current_oid = self.current_ot.optimization_id
                self._ct_model = get_table_for_optimization_id(self._ct_model_template, self._current_oid)

                if self.first_starting_ct:
                    self._current_data_hash = self.first_starting_ct.data_hash

                    existing_jobs_with_cid = self._session.query(self._ct_model)\
                        .filter(self._current_ct.data_hash == self._current_data_hash).all()

                    for job in existing_jobs_with_cid:
                        job.calculation_state = CALCULATION_RUN

                    self._session.commit()

                    if not self.any_finished_ct_with_same_id:
                        calculation_data = load_json(self.current_ct.calculation_data)

                        # data was already validated by optimization manager
                        modflowdatamodel = ModflowDataModel(calculation_data)

                        flopymodelmanager = FlopyModelManager.from_modflowdatamodel(modflowdatamodel)

                        flopymodelmanager.build_flopymodel()

                        flopymodelmanager.run_model()

                    for job in existing_jobs_with_cid:
                        job.calculation_state = CALCULATION_FINISH

                        if self.current_ct.calculation_type == OPTIMIZATION_TYPE_EVOLUTION:
                            self.current_ct.current_population += 1

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
