from copy import deepcopy

from ..tracker.registry import PIPELINE_REGISTRY


class TrackerFactory:
    def __init__(self):
        super().__init__()

    def __call__(self, args, **kwargs):
        try:
            self.name = args['name']
        except:
            pass

        if len(args) == 1:
            return PIPELINE_REGISTRY.get(self.name)()
        else:
            new_args = deepcopy(args)
            new_args.pop('name')
            return PIPELINE_REGISTRY.get(self.name)(**new_args)
