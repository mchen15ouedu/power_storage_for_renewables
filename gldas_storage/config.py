"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_file: str | Path) -> dict:
    """Load the YAML config and resolve all paths relative to the config file's folder."""
    config_file = Path(config_file).resolve()
    with open(config_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    root = config_file.parent
    paths = cfg["paths"]
    for key, value in paths.items():
        p = Path(value)
        paths[key] = p if p.is_absolute() else root / p
    cfg["root"] = root
    # The forcing-dataset block is read internally as cfg["gldas"]; newer configs
    # (e.g. NLDAS) name it "forcing" -- alias it so both styles work.
    if "gldas" not in cfg and "forcing" in cfg:
        cfg["gldas"] = cfg["forcing"]
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create the directories the pipeline writes into."""
    for key in ("raw_dir", "zonal_dir", "results_dir", "figures_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["regions_gpkg"].parent.mkdir(parents=True, exist_ok=True)
