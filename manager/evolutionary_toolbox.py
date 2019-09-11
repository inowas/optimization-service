"""
Here we build our evolutionary toolbox, that will help us to create an initial population and also generate new
populations based on the calculation results of an "old" finished population.
"""

# Imports
import numpy as np
from deap import base, creator, tools, algorithms
# Import of solvers NelderMeadSimplex
from mystic.solvers import NelderMeadSimplexSolver
# Import of the termination criteria which here is CRT, that is based on relative difference between candidates for
# termination
from mystic.termination import CandidateRelativeTolerance as CRT


# Schaffer evaluation function
def evaluate_individual(individual, funs):
    # Get our x by calling the g function with unpacked individual
    x = g_mod(individual, G_MOD_CONST)

    # Now return both function values
    return schaffer_f1(x), schaffer_f2(x)

class GAToolbox:
    def __init__(self, eta, bounds, indpb, cxpb, mutpb, weights):
        self.eta = eta
        self.bounds = bounds
        self.low = [bound[0] for bound in self.bounds]
        self.up = [bound[1] for bound in self.bounds]
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.weights = weights
        self.toolbox = None
        self.hall_of_fame = None

        self.build_toolbox()

        pass

    def build_toolbox(self):
        # Module base is used to create a container for the individuals that holds the fitness function as well as
        # later added parameters (here: pair of x1, x2)
        # First FitnessMin class is registered with weights according to optimization objective
        creator.create("FitnessMin", base.Fitness, weights=self.weights)
        # Second the Individual class having a list container with parameters and a fitness value is created
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Now toolbox with genetic algorithm functions is created
        toolbox = base.Toolbox()
        # Candidate (container for individual) is added with function make_individual that creates a pair of x1, x2
        # values; function is called with limits of x1, x2 respectively
        toolbox.register("candidate", self.make_individual, self.bounds)
        # Individuals are created by calling the candidate function which again calls the make_individual function
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.candidate)
        # Population is created putting a number of individuals in a list
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Function for mating is set to be simulated binary bounded (means parameters can't leave the given range of
        # bounds)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=self.eta, low=self.low, up=self.up)
        # Function for mutation is set to be polynomial bounded and again parameters can#t leave their bounds
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=self.eta, low=self.low, up=self.up, indpb=self.indpb)
        # Function for selection is set to be selNSGA2 which performs a multi-objective selection
        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox

    def make_individual(self):
        individual = []
        for bound in self.bounds:
            gen = list(np.random.random(bound) * np.abs(np.diff(bound)) + np.min(bound))[0]
            individual.append(gen)
        return individual

    def optimize_evolutionary(self, individuals, evaluates):
        ### EVALUATE HAS TO BE REPLACED BY SOME FUNCTION THAT TAKES INTO ACCOUNT THE CALCULATED VALUES BY WORKER ###
        # toolbox.register("evaluate", schaffer_eval)
        ############################################################################################################

        # Here we will apply evolutionary algorithms
        # As we separate the algorithm itself and the calculation, we will have to take into account our individuals
        # with their parameters as well as the function responses for all our given functions

        pass

    def optimize_linear(self):
        pass



