import importlib
import pkgutil
import sys

import abstract_rl

# the optional SB3 baseline runner imports torch (through SB3) only inside its run function
OPTIONAL = {"abstract_rl.scripts.run_baselines"}


def test_library_does_not_import_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # any `import torch` now raises ImportError
    for m in pkgutil.walk_packages(abstract_rl.__path__, "abstract_rl."):
        if m.name not in OPTIONAL:
            importlib.import_module(m.name)
