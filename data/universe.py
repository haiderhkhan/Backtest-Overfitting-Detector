"""Load the ticker universe from data/universe.yaml. Config, not code."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "universe.yaml"


def load_universe(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Returns {"name", "as_of", "start", "end", "symbols": [...]}.

    Validates the shape so a typo in the YAML fails loudly here, not
    three modules later.
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "start", "end", "symbols"):
        if key not in cfg:
            raise ValueError(f"universe config missing '{key}'")
    syms = [str(s).strip().upper() for s in cfg["symbols"]]
    if len(syms) != len(set(syms)):
        raise ValueError("universe config has duplicate symbols")
    if len(syms) < 2:
        raise ValueError("universe needs at least 2 symbols")
    return {**cfg, "symbols": syms, "start": str(cfg["start"]), "end": str(cfg["end"])}
