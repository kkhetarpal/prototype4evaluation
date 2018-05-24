from prototype4evaluation.pipeline.common import PipelinePart

class EvaluationMechanism(PipelinePart):
    def __init__(self):
        super().__init__()

    def measure_performance(self,
                            algorithm,
                            is_training,
                            iteration=None):
        """
        EvaluationMechanisms should implement this function.
        It must return some measure of performance as a float.
        Args:
            env: the Gym environment
            algorithm: the AlgorithmWrapper object to execute.
            is_training: boolean indicating if training is happening
            iteration: the iteration of the training
        Returns:
            a float with a measure of how well it performed.
        """
        raise NotImplementedError('Implement the measurement technique here')

    def get_performance(self):
        raise NotImplementedError('Return things about the performance here.')