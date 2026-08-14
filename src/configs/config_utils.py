from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError("Config file was not found")

    config = yaml.safe_load(config_path.read_text())

    return config
