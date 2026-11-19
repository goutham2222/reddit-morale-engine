"""Configuration manager loading environment variables and YAML domain configs."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict
import yaml


def find_project_root() -> Path:
    """Finds the root directory of the reddit-morale-engine package."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "configs").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@dataclass
class EngineConfig:
    project_root: Path = field(default_factory=find_project_root)
    local_data_dir: Path = field(default_factory=lambda: find_project_root() / "data")
    azure_storage_connection_string: str = ""
    azure_storage_container_name: str = "reddit-morale"
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "script:reddit-morale-engine:v1.0"
    host: str = "0.0.0.0"
    port: int = 8088

    # Loaded YAML configs
    companies: Dict[str, Any] = field(default_factory=dict)
    subreddits: Dict[str, Any] = field(default_factory=dict)
    workplace_lexicon: Dict[str, Any] = field(default_factory=dict)
    leadership_events: Dict[str, Any] = field(default_factory=dict)
    did_cohorts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "EngineConfig":
        root = find_project_root()
        # Read .env if present
        env_file = root / ".env"
        if env_file.is_file():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip())
            except Exception:
                pass

        config = cls(
            project_root=root,
            local_data_dir=Path(os.environ.get("LOCAL_DATA_DIR", str(root / "data"))),
            azure_storage_connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""),
            azure_storage_container_name=os.environ.get("AZURE_STORAGE_CONTAINER_NAME", "reddit-morale"),
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            reddit_client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
            reddit_user_agent=os.environ.get("REDDIT_USER_AGENT", "script:reddit-morale-engine:v1.0"),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8088")),
        )

        # Load YAML files
        configs_dir = root / "configs"
        if configs_dir.is_dir():
            for fname, attr in [
                ("companies.yaml", "companies"),
                ("subreddits.yaml", "subreddits"),
                ("workplace_lexicon.yaml", "workplace_lexicon"),
                ("leadership_events.yaml", "leadership_events"),
                ("did_cohorts.yaml", "did_cohorts"),
            ]:
                fpath = configs_dir / fname
                if fpath.is_file():
                    with open(fpath, "r", encoding="utf-8") as f:
                        setattr(config, attr, yaml.safe_load(f) or {})

        return config
