"""
A competition for CartPole 2018!
Algorithms are judged on how quickly they can solve CartPole
"""
from prototype4evaluation.pipeline.evaluation import EvaluationMechanism
from prototype4evaluation.tools.rl import calculate_return
from collections import deque
from collections import defaultdict
import numpy as np
import gym

class CartPole2018(EvaluationMechanism):
    environment_details = {'env_name': 'CartPole-v0',
                           'gamma': 0.99}
    last_five = deque([0]*5)
    records = defaultdict(lambda : [])
    def measure_performance(self,
                            algorithm,
                            is_training,
                            iteration=None,
                            env_rank=0):
        """
        Measures the performance of CartPole
        Args:
            algorithm:
            is_training:
            iteration:
            env_rank:

        Returns:

        """
        cp_env = gym.make(self.environment_details['env_name'])
        rollout_rewards = algorithm.do_rollout(cp_env)
        self.last_five.append(calculate_return(rollout_rewards,
                                               self.environment_details['gamma']))
        self.last_five.popleft()
        self.records[env_rank].append(np.mean(self.last_five))

    def get_performance(self):
        to_be_averaged = []
        for seed, seed_result in self.records.items():
            to_be_averaged.append(seed_result[-1])
        return np.mean(to_be_averaged), np.std(to_be_averaged)
