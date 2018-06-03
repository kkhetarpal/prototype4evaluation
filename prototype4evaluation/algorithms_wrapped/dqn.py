"""
DQN agent wrapped from the keras-rl library
"""

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from prototype4evaluation.pipeline.algorithm import AlgorithmWrapper

algorithm_hyperparameters = {
    'sequential_memory_limit': 50000,
    'nb_steps_warmup': 10,
    'target_model_update': 1e-2,
    'lr': 1e-3
}

class DQNAlgorithm(AlgorithmWrapper):
    def __init__(self,
                 environment_details,
                 seeds=None,
                 **algo_args):
        super().__init__(environment_details, seeds, **algo_args)
        self.env = gym.make(environment_details['env_name'])
        nb_actions = self.env.action_space.n
        # Next, we build a very simple model.
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())
        self.model = model
        memory = SequentialMemory(limit=algo_args['sequential_memory_limit'], window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                       nb_steps_warmup=algo_args['nb_steps_warmup'],
                       target_model_update=algo_args['target_model_update'],
                       policy=policy)
        dqn.compile(Adam(lr=algo_args['lr']), metrics=['mae'])

        self.dqn = dqn
        self.policy = policy

    def act(self, state):
        if self.dqn.processor is not None: state = self.dqn.processor.process_observation(state)
        action = self.dqn.forward(state)
        if self.dqn.processor is not None: action = self.dqn.processor.process_action(action)
        return action

    def train_step(self):
        self.dqn.fit(self.env, nb_steps=1, visualize=False, verbose=2)
