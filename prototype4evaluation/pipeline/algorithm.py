from prototype4evaluation.pipeline.common import PipelinePart

class AlgorithmWrapper(PipelinePart):
    registered = False
    def __init__(self, environment_details, seeds=None, **kwargs):
        """
        All algorithms should be wrapped in this way to work in the pipeline
        You must call `super().__init__()` to register this module.
        Args:
            environment_details: A dictionary representing the environment specification.
            seeds: Algorithm seeds
            **kwargs: Hyperparameters for the algorithm
        """
        super().__init__()
        self.environment_details = environment_details
        self.seeds = seeds
        self.save_algorithm_hyperparameters(environment_details, seeds, kwargs)
        self.registered = True

    def save_algorithm_hyperparameters(self, environment_details, seeds, kwargs):
        """
        Saves algorithm hyperparameters to disk.
        Args:
            environment_details: A dictionary representing the environment specification.
            seeds: Algorithm seeds
            **kwargs: Hyperparameters for the algorithm
        """
        pass

    def initialize_environment(self):
        raise NotImplementedError('This is where you initialize your environment')

    def train_step(self):
        """
        Performs *one training update* with the environment.
        Note that you must take care of creating the environment locally.
        This gives you flexibility of using parallelized environments and things
        like that.
        """
        raise NotImplementedError('You must implement a train step here.')

    def act(self, state):
        """
        Returns an action taken by a policy in the state.
        Args:
            state: A numpy ndarray representing a state.
        Returns:
            action: the action to execute in the environment from that state.
        """
        raise NotImplementedError('You must return an action here.')

    def do_rollout(self, env, max_steps=1000):
        """
        Executes a rollout in the environment. This is the standard
        way to record how well an agent performs in a rollout.
        Args:
            env: the environment to do the rollout in
            max_steps: the maximum number of steps before premature
                       termination
        Returns:
            a list containing rollout rewards at each step.
        """
        state = env.reset()
        rollout_rewards = []
        for _ in range(max_steps):
            action = self.act(state)
            state, reward, done, info = env.step(action)
            rollout_rewards.append(reward)
            if done:
                break
        return rollout_rewards
