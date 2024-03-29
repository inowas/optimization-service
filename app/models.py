from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, ARRAY
from datetime import datetime
from config import INITIAL_SCALAR_FITNESS


# Basically our empty database with its models gets the information to which existing database to connect
Base = declarative_base()


class OptimizationTask(Base):
    __tablename__ = "optimization_tasks"

    author = Column(String)
    project = Column(String)
    publishing_date = Column(String)
    optimization_id = Column(String, primary_key=True, unique=True)
    optimization_type = Column(String)
    optimization_state = Column(String)
    current_population = Column(Integer)
    total_population = Column(Integer)
    current_generation = Column(Integer)
    total_generation = Column(Integer)
    solution = Column(ARRAY(Float))
    scalar_fitness = Column(Float)
    data_filepath = Column(String)

    def __init__(self, author, project, optimization_id, optimization_type, optimization_state, total_population,
                 total_generation, solution, data_filepath, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.publishing_date = str(datetime.now().date())
        self.optimization_id = optimization_id
        self.optimization_type = optimization_type
        self.optimization_state = optimization_state
        self.current_population = 0
        self.total_population = total_population
        self.current_generation = 0
        self.total_generation = total_generation
        self.solution = solution
        self.scalar_fitness = INITIAL_SCALAR_FITNESS
        self.data_filepath = data_filepath


class OptimizationHistory:
    # Column definitions
    author = Column(String)
    project = Column(String)
    optimization_id = Column(String)
    generation = Column(Integer, primary_key=True, unique=True)
    scalar_fitness = Column(Float)

    def __init__(self, author, project, optimization_id, generation, scalar_fitness, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.generation = generation
        self.scalar_fitness = scalar_fitness


class CalculationTask:
    # Column definitions
    author = Column(String)
    project = Column(String)
    optimization_id = Column(String)
    calculation_id = Column(String, primary_key=True, unique=True)
    hash_id = Column(String)
    calculation_type = Column(String)
    calculation_state = Column(String)
    generation = Column(Integer)
    individual_id = Column(Integer)
    calculation_data_filepath = Column(String)
    fitness = Column(ARRAY)

    def __init__(self, author, project, optimization_id, calculation_id, hash_id, calculation_type, calculation_state,
                 generation, calculation_data_filepath, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.calculation_id = calculation_id
        self.hash_id = hash_id
        self.calculation_type = calculation_type
        self.calculation_state = calculation_state
        self.generation = generation
        self.hash_id = hash_id
        self.calculation_data_filepath = calculation_data_filepath
        self.fitness = []
