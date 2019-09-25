from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, PickleType
from sqlalchemy.dialects.postgresql import UUID
from config import INITIAL_SCALAR_FITNESS


# Basically our empty database with its models gets the information to which existing database to connect
Base = declarative_base()


class OptimizationTask(Base):
    __tablename__ = "optimization_tasks"

    author = Column(String)
    project = Column(String)
    optimization_id = Column(String, primary_key=True)
    optimization_state = Column(String)
    current_population = Column(Integer)
    total_population = Column(Integer)
    current_generation = Column(Integer)
    total_generation = Column(Integer)
    solution = Column(PickleType)
    scalar_fitness = Column(Float)
    opt_filepath = Column(String)
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


class CalculationTaskEvolutionaryOptimization(Base):
    __tablename__ = "calculation_tasks_evolutionary_optimization"

    author = Column(String)
    project = Column(String)
    optimization_id = Column(String)
    calculation_id = Column(UUID(as_uuid=True), primary_key=True, unique=True)
    calculation_state = Column(String)
    generation = Column(Integer)
    individual_id = Column(Integer)
    data_filepath = Column(String)
    calcinput_filepath = Column(String)
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


class CalculationTaskLinearOptimization(Base):
    __tablename__ = "calculation_tasks_linear_optimization"

    author = Column(String)
    project = Column(String)
    optimization_id = Column(String)
    calculation_id = Column(UUID(as_uuid=True), primary_key=True, unique=True)
    calculation_state = Column(String)
    generation = Column(Integer)
    individual_id = Column(Integer)
    data_filepath = Column(String)
    calcinput_filepath = Column(String)
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
