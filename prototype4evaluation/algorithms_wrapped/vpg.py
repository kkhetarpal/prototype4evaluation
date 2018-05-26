"""
Vanilla policy gradient algorithm wrapped from ./in_house_algorithms/vpg.py
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from pg_methods import interfaces
from pg_methods.algorithms.REINFORCE import VanillaPolicyGradient
from pg_methods.baselines import MovingAverageBaseline, NeuralNetworkBaseline
from pg_methods.networks import MLP_factory
from pg_methods.utils import experiment
from prototype4evaluation.pipeline.algorithm import AlgorithmWrapper

algorithm_hyperparameters = {
    'baseline': 'moving_average',
    'cpu_count': 2,
    'policy_lr': 0.001,
    'policy': 'multinomial'
}


class REINFORCEAlgorithm(AlgorithmWrapper):
    def __init__(self,
                 environment_details,
                 seeds=None,
                 **algo_args):
        super().__init__(environment_details, seeds, **algo_args)
        self.env = interfaces.make_parallelized_gym_env(environment_details['env_name'], 0,
                                                        algo_args['cpu_count'])
        if algo_args['baseline'] == 'moving_average':
            self.baseline = MovingAverageBaseline(0.9)
        elif algo_args['baseline'] == 'neural_network':
            self.val_approximator = MLP_factory(self.env.observation_space_info['shape'][0],
                                           [16, 16],
                                           output_size=1,
                                           hidden_non_linearity=nn.ReLU)
            self.val_optimizer = torch.optim.SGD(self.val_approximator.parameters(), lr=algo_args['value_lr'])
            self.baseline = NeuralNetworkBaseline(self.val_approximator, self.val_optimizer, bootstrap=False)
        else:
            self.baseline = None

        fn_approximator, policy = experiment.setup_policy(self.env,
                                                          hidden_non_linearity=nn.ReLU,
                                                          hidden_sizes=[16, 16])

        self.fn_approximator = fn_approximator
        self.policy = policy
        self.optimizer = torch.optim.SGD(fn_approximator.parameters(), lr=algo_args['policy_lr'])

        self.algorithm = VanillaPolicyGradient(self.env,
                                               self.policy,
                                               self.optimizer,
                                               gamma=environment_details['gamma'],
                                               baseline=self.baseline)

    def act(self, state):
        # torch 0.3 sign...
        # TODO: upgrade here
        state = Variable(self.env.observation_processor.gym2pytorch(state), volatile=True)
        action, _ = self.policy(state)
        return self.env.action_processor.pytorch2gym(action.data)


    def train_step(self):
        self.algorithm.run(1, verbose=False)

