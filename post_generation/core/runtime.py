from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _find_post_generation_root(current_file: PathLike) -> Path:
    file_path = Path(current_file).resolve()
    candidates = [file_path.parent]
    candidates.extend(file_path.parents)

    for candidate in candidates:
        if candidate.name == "post_generation":
            return candidate

    raise ValueError(
        f"Could not resolve post_generation root from path: {file_path}"
    )


def ensure_post_generation_paths(current_file: PathLike) -> Path:
    """Add both workspace root and post_generation root to sys.path."""
    project_root = _find_post_generation_root(current_file)
    workspace_root = project_root.parent

    for path in (workspace_root, project_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return project_root
