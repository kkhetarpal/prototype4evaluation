import gym
from prototype4evaluation.pipeline.algorithm import AlgorithmWrapper

class RandomAlgorithm(AlgorithmWrapper):
    def __init__(self, environment_details, seeds=None, **kwargs):
        """
        All algorithms should be wrapped in this way to work in the pipeline
        You must call `super().__init__()` to register this module.
        Args:
            environment_details: A dictionary representing the environment specification.
            seeds: Algorithm seeds
            **kwargs: Hyperparameters for the algorithm
        """
        super().__init__(environment_details)
        self.env = gym.make(self.environment_details['env_name'])

    def train_step(self):
        return 0

    def act(self, state):
        return self.env.action_space.sample()

