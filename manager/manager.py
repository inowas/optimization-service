from time import sleep
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.engine import create_engine

# Database to our optimization postgres
DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"

db = SQLAlchemy()
db.create_engine(DATABASE_URL)


class BaseModel(db.Model):
    """Base data model for all objects"""
    __abstract__ = True

    def __init__(self, **args):
        super().__init__(**args)

    def __repr__(self):
        """Define a base way to print models"""
        return '%s(%s)' % (self.__class__.__name__, {
            column: value for column, value in self._to_dict().items()
        })


class CalculationTask(BaseModel, db.Model):
    __tablename__ = "calculation_tasks"

    author = db.Column(db.String)
    # Column for project
    project = db.Column(db.String)
    # Column for calculation_id
    calculation_id = db.Column(db.String)
    # Column for model_id
    model_id = db.Column(db.String)
    # Column for optimization_id
    optimization_id = db.Column(db.String, primary_key=True)
    # Column for type
    type = db.Column(db.String)
    # Column for version
    version = db.Column(db.String)
    # Column for optimization
    optimization = db.Column(db.PickleType)
    # Column for data
    data = db.Column(db.PickleType)

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


def run():
    while True:
        print('No jobs, sleeping for 10 seconds')
        sleep(10)
        continue


if __name__ == '__main__':
    run()
