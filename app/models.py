from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, PickleType
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4


# Add the created app to our database
# Basically our empty database with its models gets the information to which existing database to connect
Base = declarative_base()


class OptimizationTask(Base):
    __tablename__ = "optimization_tasks"

    author = Column(String)
    # Column for project
    project = Column(String)
    # Column for optimization_id
    optimization_id = Column(String, primary_key=True)
    # Column for type
    optimization_type = Column(String)
    # Column for optimization
    optimization = Column(PickleType)
    # Column for data
    # data = db.Column(db.PickleType)

    def __init__(self, author, project, optimization_id, optimization_type, optimization, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        # self.calculation_id = calculation_id
        # self.model_id = model_id
        self.optimization_id = optimization_id
        self.optimization_type = optimization_type
        # self.version = version
        self.optimization = optimization
        # self.data = data


class CalculationTask(Base):
    __tablename__ = "calculation_tasks"

    author = Column(String)
    # Column for project
    project = Column(String)
    # Column for optimization_id
    optimization_id = Column(String)
    # Column for calculation_id
    calculation_id = Column(UUID(as_uuid=True), primary_key=True, unique=True, default=uuid4)
    # Column for type
    calculation_type = Column(String)
    # Column for calculation parameters
    calculation_parameters = Column(PickleType)

    def __init__(self, author, project, optimization_id, calculation_type, calculation_parameters, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.calculation_type = calculation_type
        self.calculation_parameters=calculation_parameters
