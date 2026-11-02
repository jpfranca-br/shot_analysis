import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


class AppConfig(BaseModel):
    store_videos: bool = False
    store_uploaded_videos: bool = False
    store_rendered_videos: bool = False

    @property
    def keep_uploaded(self) -> bool:
        return bool(self.store_videos or self.store_uploaded_videos)

    @property
    def keep_rendered(self) -> bool:
        return bool(self.store_videos or self.store_rendered_videos)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load application configuration from config.json (defaulting to not storing videos).
    Environment variable ANALYZER_STORE_VIDEOS ('true'/'1'/'yes') can also override.
    """
    path = Path(
        config_path
        or os.environ.get("ANALYZER_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    )
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}

    cfg = AppConfig(**data)

    env_override = os.environ.get("ANALYZER_STORE_VIDEOS")
    if env_override is not None:
        val = env_override.strip().lower() in {"1", "true", "yes", "on"}
        cfg.store_videos = val
        cfg.store_uploaded_videos = val
        cfg.store_rendered_videos = val

    return cfg
