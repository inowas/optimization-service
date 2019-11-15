import os.path
import sys
from time import sleep
from flopyAdapter import ModflowDataModel, FlopyModelManager
# from flopyAdapter import FlopyDataModel
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from helper_functions import load_json, get_table_for_optimization_id  # noqa: E402
from db import Session  # noqa: E402
from models import CalculationTask, OptimizationTask  # noqa: E402
from config import OPTIMIZATION_RUN, CALCULATION_START, CALCULATION_RUN, CALCULATION_FINISH, \
    OPTIMIZATION_TYPE_EVOLUTION  # noqa: E402

MISSING_DATA_VALUE = -9999


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
        self.current_calculation_id2 = None
        self.current_ct = None

    def reset_temporary_attributes(self):
        self.current_optimization_id = None
        self.current_calculation_id2 = None
        self.current_ct = None

    def query_first_starting_calculation_task(self) -> Session.query:
        """ Return first calculation task that has not yet been started

        Returns:
            query - first optimization task in list

        """
        return self.session.query(self.current_ct)\
            .filter(self.current_ct.calculation_state == CALCULATION_START).first()

    def query_calculation_task_with_id(self) -> Session.query:

        return self.session.query(self.current_ct).\
            filter(self.current_ct.calculation_id == self.current_calculation_id2).first()

    def query_current_optimization_task(self) -> Session.query:

        return self.session.query(self.ot).\
            filter(self.ot.optimization_id == self.current_optimization_id).first()

    def run(self):
        while True:

            # todo implement worktask based on hash id that checks if there's already a job calculating with the hashid
            # todo and instead waits for it to be finished and then reuses the same results
            running_optimization_task = self.session.query(self.ot)\
                .filter(self.ot.optimization_state == OPTIMIZATION_RUN).first()

            if running_optimization_task:

                self.current_optimization_id = running_optimization_task.optimization_id
                self.current_ct = get_table_for_optimization_id(self.ct, self.current_optimization_id)

                new_calculation_task = self.query_first_starting_calculation_task()

                if new_calculation_task:
                    self.current_calculation_id2 = new_calculation_task.calculation_id2

                    jobs_with_same_calculation_id2 = self.session.query(self.current_ct)\
                        .filter(self.current_ct.calculation_id2 == self.current_calculation_id2).all()

                    for job in jobs_with_same_calculation_id2:
                        job.calculation_state = CALCULATION_RUN

                    # new_calculation_task.calculation_state = CALCULATION_RUN

                    self.session.commit()

                    # todo check if already existing and if not start this

                    calculation_data = load_json(new_calculation_task.calculation_data)

                    # Build model
                    modflowdaatmodel = ModflowDataModel.from_data(calculation_data)

                    flopymodelmanager = FlopyModelManager.from_modflowdatamodel(modflowdaatmodel)

                    flopymodelmanager.build_flopymodel()

                    flopymodelmanager.run_model()

                    # flopy_data_model = FlopyDataModel(version=calculation_data["version"],
                    #                                   data=calculation_data["data"],
                    #                                   uuid=calculation_data["optimization_id"])
                    #
                    # flopy_data_model.add_wells(objects=calculation_data["optimization"]["objects"])
                    #
                    # flopy_data_model.build_flopy_models()
                    #
                    # flopy_data_model.run_models()

                    # fitness = flopy_data_model.get_fitness(objectives=calculation_data["optimization"]["objectives"],
                    #                                        constraints=calculation_data["optimization"]["constraints"],
                    #                                        objects=calculation_data["optimization"]["objects"])

                    # todo else if existing already a finished job take its fitness instead!

                    for job in jobs_with_same_calculation_id2:
                        # job.scalar_fitness = fitness
                        job.calculation_state = CALCULATION_FINISH

                        if running_optimization_task.optimization_type == OPTIMIZATION_TYPE_EVOLUTION:
                            running_optimization_task.current_population += 1

                    # new_calculation_task.scalar_fitness = fitness
                    # new_calculation_task.calculation_state = CALCULATION_FINISH

                    self.session.commit()

                    continue


if __name__ == '__main__':
    sleep(10)

    worker_manager = WorkerManager(
        session=Session,
        optimization_task=OptimizationTask,
        calculation_task=CalculationTask
    )

    worker_manager.run()
