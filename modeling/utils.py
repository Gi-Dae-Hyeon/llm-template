from importlib import import_module as _import_module


def import_module(module: str):
    from_, import_ = module.rsplit(".", 1)
    return getattr(_import_module(from_), import_)