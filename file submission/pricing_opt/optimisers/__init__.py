import importlib
from pathlib import Path

# always import linear first
importlib.import_module(f"{__name__}.linear")

_pkg_path = Path(__file__).parent
for f in _pkg_path.iterdir():
    if f.stem in {"__init__", "linear"} or not f.suffix == ".py":
        continue
    importlib.import_module(f"{__name__}.{f.stem}")
