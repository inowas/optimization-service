"""
Here we build our evolutionary toolbox, that will help us to create an initial population and also generate new
populations based on the calculation results of an "old" finished population.
"""

# Imports
from typing import List, Tuple
import numpy as np
from deap import base, creator, tools, algorithms
from copy import deepcopy
from mystic.solvers import NelderMeadSimplexSolver
from mystic.termination import CandidateRelativeTolerance as CRT


class EAToolbox:
    def __init__(self,
                 eta: float,
                 bounds: List[List[float]],
                 indpb: float,
                 cxpb: float,
                 mutpb: float,
                 weights: Tuple[int, int]):
        # Usecase related parameters
        self.eta = eta
        self.bounds = bounds
        self.low = [bound[0] for bound in self.bounds]
        self.up = [bound[1] for bound in self.bounds]
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.weights = weights
        # Deap classes and methods
        self.fitness_class = base.Fitness
        self.mate_method = tools.cxSimulatedBinaryBounded
        self.mutate_method = tools.mutPolynomialBounded
        self.select_method = tools.selNSGA2
        # Deap main structure
        self.hall_of_fame = tools.ParetoFront()
        self.default_individual = None
        self.toolbox = None


        # Create our default individual which is used by our toolbox
        self.build_default_individual()
        # Create our ga toolbox which is used for further ga optimization
        self.build_toolbox()

    def build_default_individual(self):
        """
        Here we create our individual template that consists of the individuals parameters as well as the fitness
        attribute that holds an array of individual fitnesses and corresponding weights
        Args:
            self - holds the weights we need for our base function weighting
        Returns:
            None - the base individual is written on self.default_individual
        """
        # Module base is used to create a container for the individuals that holds the fitness function as well as
        # later added parameters (here: pair of x1, x2)
        # First FitnessMin class is registered with weights according to optimization objective
        creator.create("FitnessMin", self.fitness_class, weights=self.weights)
        # Second the Individual class having a list container with parameters and a fitness value is created
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.default_individual = creator.Individual

    def make_candidate(self) -> List[float]:
        """ Function to create our candidate based on the bounds given for every parameter. The difference to the individual is
        that a candidate only holds parameter values and has no fitness attribute.

        Args:
            self - holds the bounds with which parameters are randomly chosen

        Returns:
            None - the candidate is used to create an individual in a way that it's the parameter holder

        """
        individual = []

        for bound in self.bounds:
            # Scale individual
            gen = list(np.random.random(1) * np.abs(np.diff(bound)) + np.min(bound))[0]
            individual.append(gen)

        return individual

    def build_toolbox(self) -> None:
        """ Function to build our toolbox that mainly takes into account the parsed ga parameters from our upload

        Args:
            self - holds the functions make_individual, bounds,

        Returns:
            None - the base individual is written on self.default_individual

        """

        # Now toolbox with genetic algorithm functions is created
        toolbox = base.Toolbox()
        toolbox.register("candidate", self.make_candidate)
        toolbox.register("individual", tools.initIterate, deepcopy(self.default_individual), toolbox.candidate)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Function for mating is set to be simulated binary bounded (means parameters can't leave the given range of
        # bounds)
        toolbox.register("mate", self.mate_method, eta=self.eta, low=self.low, up=self.up)
        # Function for mutation is set to be polynomial bounded and again parameters can#t leave their bounds
        toolbox.register("mutate", self.mutate_method, eta=self.eta, low=self.low, up=self.up, indpb=self.indpb)
        # Function for selection is set to be selNSGA2 which performs a multi-objective selection
        toolbox.register("select", self.select_method)

        self.toolbox = toolbox

    def make_population(self,
                        population_size: int) -> List[List[float]]:
        return self.toolbox.population(population_size)

    def evaluate_finished_calculations(self,
                                       individuals: List[dict]) -> List[List[float]]:
        """ Function the complete the individuals with their fitness in form of the function returns

        Args:
            individuals: a list of dicts that define each individual - by parameters and by function evaluations

        Returns:
            population: a population of individuals that have a fitness on them based on the function returns (tuple of
            f1(individual), f2(individual), f3...)

        """
        population = []
        for ind in individuals:
            individual = self.default_individual()

            individual.extend(ind["ind_genes"])

            individual.fitness.values = tuple(ind["functions"][fun] for fun in ind["functions"])
            # individual.fitness.valid = True

            population.append(individual)

        return population

    def select_best_individuals(self,
                                population):
        """ Function wrapper to run the select function from our ga toolbox and also add those selected individuals to
        the hall of fame

        Args:
            population: list of individuals with certain parameters

        Returns:
            population: a ordered selection from the population based on the fitness

        """
        population = self.toolbox.select(population,
                                         k=len(population))

        self.hall_of_fame.update(population)

        return population

    def select_first_of_hall_of_fame(self):
        """ Function wrapper to return the first of equally optimal solutions of the paretofront-halloffame

        Returns:
            hall_of_fame: first individual (hall of fame solutions can be seen equally for a paretofront type of hall
            of fame

        """

        return self.hall_of_fame[0]

    def optimize_evolutionary(self,
                              individuals: List[dict]) -> List[List[float]]:
        """ Function to optimize a list of individuals with an genetic algorithm from the deap library

        Args:
            self - holds the toolbox
            individuals - dictionaries that hold the genes of the individual (parameters) and the evaluate/function values

        Returns:
            None - the base individual is written on self.default_individual

        """
        population = self.evaluate_finished_calculations(individuals=individuals)

        population = self.select_best_individuals(population=population)

        population = algorithms.varAnd(population=population,
                                       toolbox=self.toolbox,
                                       cxpb=self.cxpb,
                                       mutpb=self.mutpb)

        return population

    def optimize_linear(self,
                        solution: List[float],
                        function) -> List[float]:
        """ Function to optimize one solution linear by using the mystic library

        Args:
            solution: the initial solution that the solver starts with
            function: the callback function that sends out the task to the database, awaits the result and takes it
            back in

        Returns:
            solution: a linear optimized solution

        """
        solver = NelderMeadSimplexSolver(dim=len(self.weights))

        solver.SetInitialPoints(x0=solution)

        solver.SetTermination(CRT())

        solver.Solve(function)

        return solver.Solution()
