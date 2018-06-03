from prototype4evaluation.pipeline.common import BasePipeline

class Pipeline(BasePipeline):
    def __init__(self,
                 algorithm_constructor,
                 algorithm_arguments,
                 evaluation_constructor,
                 evaluation_arguments,
                 repeats=5):
        super().__init__()
        self.algorithm_constructor = algorithm_constructor
        self.algorithm_arguments = algorithm_arguments
        self.evaluation_constructor = evaluation_constructor
        self.evaluation_arguments = evaluation_arguments

    def _run_parallel(self):
        raise NotImplementedError('Future work.')

    def run(self, training_steps):
        evaluation = self.evaluation_constructor(**self.evaluation_arguments)

        for rank in range(3):
            print('Replicate {}/{}'.format(rank+1, 3))
            algorithm = self.algorithm_constructor(evaluation.environment_details,
                                                   **self.algorithm_arguments)

            for i in range(training_steps):
                algorithm.train_step()
                evaluation.measure_performance(algorithm,
                                               is_training=True,
                                               iteration=i,
                                               env_rank=rank)

        return evaluation

