import importlib
import pkgutil

import pytest

import abstract_rl

# every module, including abstract_rl.scripts.*, must import without starting a training run
# (scripts guard their entry point with ``if __name__ == "__main__"``)
MODULES = [m.name for m in pkgutil.walk_packages(abstract_rl.__path__, "abstract_rl.")]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
