"""Preserve legacy imports as aliases of their canonical implementation modules."""
import importlib
import runpy
import sys


def redirect(module_name, target):
    """Keep module globals, private helpers and mock patch targets shared."""
    if module_name == "__main__":
        runpy.run_module(target, run_name="__main__")
    else:
        sys.modules[module_name] = importlib.import_module(target)
