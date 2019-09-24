from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, PickleType
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from config import INITIAL_SCALAR_FITNESS


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
    optimization_state = Column(String)
    # Column for current runs
    current_population = Column(Integer)
    # Column for total runs
    total_population = Column(Integer)
    # Column for current generation
    current_generation = Column(Integer)
    # Column for total generations
    total_generation = Column(Integer)
    # Column for calculated solution
    solution = Column(PickleType)
    # Column for fitness of optimization task after being finished
    scalar_fitness = Column(Float)
    # Column for optimization filepath
    opt_filepath = Column(String)
    # Column for data filepath
    data_filepath = Column(String)

    def __init__(self, author, project, optimization_id, optimization_state, total_population, total_generation,
                 solution, opt_filepath, data_filepath, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.optimization_state = optimization_state
        self.current_population = 0
        self.total_population = total_population
        self.current_generation = 0
        self.total_generation = total_generation
        self.solution = solution
        self.fitness = INITIAL_SCALAR_FITNESS
        self.opt_filepath = opt_filepath
        self.data_filepath = data_filepath


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
    calculation_state = Column(String)
    # Column for Generation number
    generation = Column(String)
    # Column for individual (in range(popsize))
    individual_id = Column(String)
    # Column for data filepath
    data_filepath = Column(String)
    # Column for calculation_input filepath
    calcinput_filepath = Column(String)
    # Column for calculation_output filepath
    calcoutput_filepath = Column(String)

    def __init__(self, author, project, optimization_id, calculation_id, calculation_state, generation, individual_id,
                 data_filepath, calcinput_filepath, calcoutput_filepath, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.calculation_id = calculation_id
        self.calculation_state = calculation_state
        self.generation = generation
        self.individual_id = individual_id
        self.data_filepath = data_filepath
        self.calcinput_filepath = calcinput_filepath
        self.calcoutput_filepath = calcoutput_filepath
