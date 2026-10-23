import subprocess
import sys
import textwrap

# the optional SB3 baseline runner imports torch (through SB3) and Gymnasium only inside its run function
OPTIONAL = {"abstract_rl.scripts.run_baselines"}

# run in a fresh interpreter: modules imported by conftest or other tests would otherwise hide a missing import
SCRIPT = textwrap.dedent(
    """
    import importlib
    import pkgutil
    import sys

    sys.modules["torch"] = None  # any `import torch` now raises ImportError
    import abstract_rl

    optional = {optional!r}
    for m in pkgutil.walk_packages(abstract_rl.__path__, "abstract_rl."):
        if m.name not in optional:
            importlib.import_module(m.name)
    loaded = [name for name in ("gymnasium", "gym") if name in sys.modules]
    assert not loaded, f"core modules import {{loaded}}"
    """
)


def test_library_imports_neither_torch_nor_gymnasium():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(optional=sorted(OPTIONAL))], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
