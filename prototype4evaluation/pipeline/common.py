
class PipelinePart(object):
    def __init__(self):
        raise NotImplementedError('Any parts of the pipeline must inherit this.')


class BasePipeline(PipelinePart):
    def __init__(self):
        super().__init__()
        raise NotImplementedError('You must implement this class')

