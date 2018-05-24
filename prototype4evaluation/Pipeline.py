from prototype4evaluation.pipeline.common import BasePipeline

class Pipeline(BasePipeline):
    def __init__(self,
                 algorithm_constructor,
                 algorithm_arguments,
                 evaluation_constructor,
                 evaluation_arguments,
                 repeats=5):
        super().__init__()
        self.algorithm_constructor = evaluation_constructor
        self.algorithm_arguments = evaluation_arguments
        self.evaluation_constructor = evaluation_constructor
        self.evaluation_arguments = evaluation_arguments

    def _run_parallel(self):
        raise NotImplementedError('Future work.')

    def start(self):
        """
        Creates an instance of the algorithm
        Returns:
            An Algorithm and an Evaluation Scheme.
        """
        # abstracted out so that we can eventually do this in parallel.
        alg = self.algorithm_constructor(**self.algorithm_arguments)
        eval = self.evaluation_constructor(**self.evaluation_arguments)
        return alg, eval

    def run(self, training_steps):
        algorithm, evaluation = self.start()

        for rank in range(3):
            for i in range(training_steps):
                algorithm.train_step()
                evaluation.measure_performance(algorithm,
                                               is_training=True,
                                               iteration=i,
                                               env_rank=rank)

        return evaluation.get_performance()

