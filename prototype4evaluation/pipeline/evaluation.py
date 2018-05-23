from prototype4evaluation.pipeline.common import PipelinePart

class EvaluationMechanism(PipelinePart):
    def __init__(self):
        super().__init__()

    def measure_performance(self, env, algorithm, training_performance=None):
        """
        EvaluationMechanisms should implement this function.
        It must return some measure of performance as a float.
        Args:
            env: the Gym environment
            algorithm: the AlgorithmWrapper object to execute.
            training_performance: Information regarding the training performance
                                  of the algorithm.
        Returns:
            a float with a measure of how well it performed.
        """
        raise NotImplementedError('Implement the measurement technique here')

