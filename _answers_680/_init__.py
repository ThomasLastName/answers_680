
from importlib import import_module

__all__ = [f"answers_week_{j+1}" for j in range(15)]

for module_name in __all__:
    globals()[module_name] = import_module(module_name)
