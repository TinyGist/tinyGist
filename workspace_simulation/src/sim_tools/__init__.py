from importlib import import_module

from src.sim_tools.definitions import UTILS_LAZY_IMPORTS

__all__ = list(UTILS_LAZY_IMPORTS.keys())


def __getattr__(name):
    if name not in UTILS_LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = UTILS_LAZY_IMPORTS[name]
    attr = getattr(import_module(module_name), attr_name)
    globals()[name] = attr
    return attr
