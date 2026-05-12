"""
Auto-register all custom run spec functions in this directory.

Importing this package imports every *_run_specs.py module, which triggers
their @run_spec_function decorators and registers them into HELM's global
run-spec registry.
"""

import importlib
import pathlib

_pkg_dir = pathlib.Path(__file__).parent

for _f in sorted(_pkg_dir.glob("*_run_specs.py")):
    importlib.import_module(f"{__name__}.{_f.stem}")
