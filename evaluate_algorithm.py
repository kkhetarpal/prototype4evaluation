from prototype4evaluation.tasks.CartPole2018.cartpole2018 import CartPole2018
from prototype4evaluation.algorithms_wrapped.random import RandomAlgorithm
from prototype4evaluation.algorithms_wrapped.vpg import REINFORCEAlgorithm, algorithm_hyperparameters
from prototype4evaluation.algorithms_wrapped.dqn import DQNAlgorithm, algorithm_hyperparameters
from prototype4evaluation.Pipeline import Pipeline

if __name__ == '__main__':

    algorithm_constructor = DQNAlgorithm
    algorithm_arguments = algorithm_hyperparameters
    evaluation_constructor = CartPole2018
    evaluation_arguments = {}
    pipeline = Pipeline(algorithm_constructor,
                        algorithm_arguments,
                        evaluation_constructor,
                        evaluation_arguments,
                        repeats=5)

    evaluation_results = pipeline.run(100)
    print(evaluation_results.records) # the raw curves
    print(evaluation_results.get_performance()) # the summary