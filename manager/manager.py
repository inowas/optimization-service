from sqlalchemy import Column, String, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
from time import sleep


# Database to our optimization postgres
DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"


#create declarative_base instance
Base = declarative_base()


class OptimizationTask(Base):
    __tablename__ = "optimization_tasks"

    author = Column(String)
    # Column for project
    project = Column(String)
    # # Column for calculation_id
    # calculation_id = db.Column(db.String)
    # # Column for model_id
    # model_id = db.Column(db.String)
    # Column for optimization_id
    optimization_id = Column(String, primary_key=True)
    # Column for type
    type = Column(String)
    # Column for version
    # version = db.Column(db.String)
    # Column for optimization
    optimization = Column(PickleType)
    # Column for data
    # data = db.Column(db.PickleType)

    def __init__(self, author, project, optimization_id, opt_type, optimization, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        # self.calculation_id = calculation_id
        # self.model_id = model_id
        self.optimization_id = optimization_id
        self.type = opt_type
        # self.version = version
        self.optimization = optimization
        # self.data = data


class CalculationTask(Base):
    __tablename__ = "calculation_tasks"

    author = Column(String)
    # Column for project
    project = Column(String)
    # Column for calculation_id
    calculation_id = Column(String)
    # Column for model_id
    model_id = Column(String)
    # Column for optimization_id
    optimization_id = Column(String, primary_key=True)
    # Column for type
    type = Column(String)
    # Column for version
    version = Column(String)
    # Column for optimization
    optimization = Column(PickleType)
    # Column for data
    data = Column(PickleType)

    def __init__(self, author, project, calculation_id, model_id, optimization_id, opt_type, version, optimization, data, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.calculation_id = calculation_id
        self.model_id = model_id
        self.optimization_id = optimization_id
        self.type = opt_type
        self.version = version
        self.optimization = optimization
        self.data = data


#creates a create_engine instance at the bottom of the file
engine = create_engine(DATABASE_URL)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

Base.query = db_session.query_property()

Base.metadata.create_all(engine)


def run():
    while True:
        current_task = OptimizationTask.query.first()
        print(f'Current job: {current_task.optimization_id}. sleeping for 10 seconds')
        sleep(10)
        continue


if __name__ == '__main__':
    run()
