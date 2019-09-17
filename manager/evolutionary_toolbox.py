"""
Here we build our evolutionary toolbox, that will help us to create an initial population and also generate new
populations based on the calculation results of an "old" finished population.
"""

# Imports
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Tuple
# from copy import deepcopy
# Import of solvers NelderMeadSimplex
from mystic.solvers import NelderMeadSimplexSolver
# Import of the termination criteria which here is CRT, that is based on relative difference between candidates for
# termination
from mystic.termination import CandidateRelativeTolerance as CRT

# HOF_SIZE = 10

# # Schaffer evaluation function
# def evaluate_individual(individual, fun_evals):
#     # Get our x by calling the g function with unpacked individual
#     x = g_mod(individual, G_MOD_CONST)
#
#     # Now return both function values
#     return (fun(x) for fun in funs) # schaffer_f1(x), schaffer_f2(x)


class GAToolbox:
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
        """
        Here we create our candidate based on the bounds given for every parameter. The difference to the individual is
        that a candidate only holds parameter values and has no fitness attribute.
        Args:
            self - holds the bounds with which parameters are randomly chosen
        Returns:
            None - the candidate is used to create an individual in a way that it's the parameter holder
        """
        # Individual holds a list of parameters
        individual = []
        # For every bound
        for bound in self.bounds:
            # We scale a random number (which is uniform distributed between 0 and 1) up to the range of the individual
            # bound. For example of we draw a random number 0.5 and our parameter has a range of [0, 10] this would
            # result in a candidate parameter of 5.
            gen = list(np.random.random(1) * np.abs(np.diff(bound)) + np.min(bound))[0]
            individual.append(gen)
        return individual

    def build_toolbox(self):
        """
        Args:
            self - holds the functions make_individual, bounds,
        Returns:
            None - the base individual is written on self.default_individual
        """

        # Now toolbox with genetic algorithm functions is created
        toolbox = base.Toolbox()
        # Candidate (container for individual) is added with function make_individual that creates a pair of x1, x2
        # values; function is called with limits of x1, x2 respectively
        toolbox.register("candidate", self.make_candidate, self.bounds)
        # Individuals are created by calling the candidate function which again calls the make_individual function
        toolbox.register("individual", tools.initIterate, deepcopy(self.default_individual), toolbox.candidate)
        # Population is created putting a number of individuals in a list
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

    def optimize_evolutionary(self,
                              individuals: List[dict]) -> List[List[float]]:
        """
        Args:
            self - holds the toolbox
            individuals - dictionaries that hold the genes of the individual (parameters) and the evaluate/function values
        Returns:
            None - the base individual is written on self.default_individual
        """

        # We have to build a population here that works with the toolbox
        population = []
        # Loop over individuals
        for ind in individuals:
            # Create a template for one individual
            individual = self.default_individual()
            # Add parameters
            individual.extend(ind["ind_genes"])
            # Add fitness values
            individual.fitness = (ind["functions"][fun] for fun in ind["functions"])
            # Add individual to population
            population.append(individual)

        population, _ = algorithms.eaSimple(toolbox=self.toolbox,
                                            population=population,
                                            cxpb=self.cxpb,
                                            mutpb=self.mutpb,
                                            ngen=1,
                                            halloffame=self.hall_of_fame,
                                            verbose=False)

        # Now we can select the top n individuals

        ### EVALUATE HAS TO BE REPLACED BY SOME FUNCTION THAT TAKES INTO ACCOUNT THE CALCULATED VALUES BY WORKER ###
        # toolbox.register("evaluate", schaffer_eval)
        ############################################################################################################

        # Here we will apply evolutionary algorithms
        # As we separate the algorithm itself and the calculation, we will have to take into account our individuals
        # with their parameters as well as the function responses for all our given functions.
        # First we have to build a population out of our individuals and their evaluates, which means that they
        # fit into our base

        return population

    def optimize_linear(self):
        pass



