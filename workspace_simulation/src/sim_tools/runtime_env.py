import os
from pathlib import Path
import tempfile


def configure_runtime_cache_dirs() -> Path:
    """Set writable cache directories before plotting libraries are imported."""
    cache_root = _cache_root()
    matplotlib_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"

    _set_writable_dir("MPLCONFIGDIR", matplotlib_dir)
    _set_writable_dir("XDG_CACHE_HOME", xdg_cache_dir)
    return cache_root


def _cache_root() -> Path:
    configured = os.environ.get("WORKSPACE_SIM_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
    else:
        root = Path(tempfile.gettempdir()) / "workspace_sim_cache"

    if _prepare_dir(root):
        return root

    fallback = Path(tempfile.gettempdir()) / "workspace_sim_cache"
    _prepare_dir(fallback)
    return fallback


def _set_writable_dir(env_name: str, default_path: Path):
    current_value = os.environ.get(env_name)
    if current_value and _prepare_dir(Path(current_value).expanduser()):
        return

    if _prepare_dir(default_path):
        os.environ[env_name] = str(default_path)


def _prepare_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return path.is_dir() and os.access(path, os.W_OK)
