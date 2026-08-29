from contextlib import contextmanager

import torch


@contextmanager
def isolated_model_initialization_rng():
    """Keep CPU model-initialization seeds out of runtime RNG streams."""

    with torch.random.fork_rng(devices=[]):
        yield


class ModelRegistry(dict):
    def register(self, name, model_class=None):
        def decorator(cls):
            self[name] = cls
            return cls

        if model_class is not None:
            return decorator(model_class)
        return decorator

    def create(self, name, *args, **kwargs):
        if name not in self:
            raise NotImplementedError(f"Model {name} is not implemented yet, only support {self.keys()}")
        return self[name](*args, **kwargs)


class CriteriaRegistry(dict):
    def register(self, name, criteria_class=None):
        def decorator(cls):
            self[name] = cls
            return cls

        if criteria_class is not None:
            return decorator(criteria_class)
        return decorator


MODEL_REGISTRY = ModelRegistry()
CRITERIA_REGISTRY = CriteriaRegistry()
