from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, Date, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from config import INITIAL_SCALAR_FITNESS


# Basically our empty database with its models gets the information to which existing database to connect
Base = declarative_base()


class OptimizationTask(Base):
    __tablename__ = "optimization_tasks"

    author = Column(String)
    project = Column(String)
    publishing_date = Column(String)
    optimization_id = Column(String, primary_key=True)
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


class OptimizationHistory:  # (Base)
    # __tablename__ = "optimization_progress"

    author = Column(String)
    project = Column(String)
    optimization_id = Column(String)
    generation = Column(Integer, primary_key=True)
    scalar_fitness = Column(Float)

    def __init__(self, author, project, optimization_id, generation, scalar_fitness, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.generation = generation
        self.scalar_fitness = scalar_fitness


class CalculationTask:  # EvolutionaryOptimization (Base)
    # __tablename__ = "calculation_tasks"  # _evolutionary_optimization

    author = Column(String)
    project = Column(String)
    optimization_id = Column(String)
    calculation_id = Column(String, primary_key=True, unique=True)  # UUID(as_uuid=False)
    calculation_type = Column(String)
    calculation_state = Column(String)
    generation = Column(Integer)
    individual_id = Column(Integer)
    data_filepath = Column(String)
    calcinput_filepath = Column(String)
    calcoutput_filepath = Column(String)

    def __init__(self, author, project, optimization_id, calculation_id, calculation_type, calculation_state,
                 generation, individual_id, data_filepath, calcinput_filepath, calcoutput_filepath, **args):
        super().__init__(**args)

        self.author = author
        self.project = project
        self.optimization_id = optimization_id
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.calculation_state = calculation_state
        self.generation = generation
        self.individual_id = individual_id
        self.data_filepath = data_filepath
        self.calcinput_filepath = calcinput_filepath
        self.calcoutput_filepath = calcoutput_filepath
