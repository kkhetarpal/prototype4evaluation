from prototype4evaluation.tasks.CartPole2018.cartpole2018 import CartPole2018
from prototype4evaluation.in_house_algorithms.random import RandomAlgorithm
from prototype4evaluation.Pipeline import Pipeline

if __name__ == '__main__':

    algorithm_constructor = RandomAlgorithm
    algorithm_arguments = {}
    evaluation_constructor = CartPole2018
    evaluation_arguments = {}
    pipeline = Pipeline(algorithm_constructor,
                        algorithm_arguments,
                        evaluation_constructor,
                        evaluation_arguments,
                        repeats=5)

    result = pipeline.run(50)
    print(result)