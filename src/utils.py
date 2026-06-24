"""
darktrace_phase1/src/utils.py
Shared helpers: config loading (YAML or JSON), logging, dir creation, seeding.
Kept dependency-light so the pipeline runs in a basic Python environment.
"""
from __future__ import annotations
import json, logging, os, random
import numpy as np

# --- Kaggle / newer-pandas compatibility -----------------------------------
# Some environments (notably Kaggle) default pandas to the PyArrow string
# backend, which yields arrow-backed extension arrays that scikit-learn's
# train_test_split cannot index ("only integer scalar arrays can be converted
# to a scalar index"). Force the classic NumPy backend for predictable behaviour.
try:  # pragma: no cover - environment dependent
    import pandas as _pd
    _pd.set_option("mode.string_storage", "python")
    # only set if the option exists in this pandas version
    if "future.infer_string" in [k for k in _pd._config.config._registered_options]:
        _pd.set_option("future.infer_string", False)
except Exception:
    pass
# ---------------------------------------------------------------------------


def load_config(path):
    """Load YAML if PyYAML is available, else fall back to JSON."""
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            # minimal fallback: configs are also provided as .json
            alt = path.rsplit(".", 1)[0] + ".json"
            if os.path.exists(alt):
                with open(alt) as f:
                    return json.load(f)
            raise
    return json.loads(text)


def ensure_dirs(cfg):
    for key in ("tables", "figures", "logs"):
        os.makedirs(cfg["paths"][key], exist_ok=True)
    os.makedirs(cfg["paths"].get("processed", "data/processed"), exist_ok=True)


def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log")); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
