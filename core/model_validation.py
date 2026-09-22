# -*- coding: utf-8 -*-
"""Validation helpers for configured local model files."""

from pathlib import Path
from typing import Iterable, List, Tuple


class ModelValidationError(RuntimeError):
    """Raised when one or more configured model files are unavailable."""


def resolve_config_relative_path(path_value: str, config_path: str) -> Path:
    """Resolve a model path relative to the configuration file."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config_path).resolve().parent / path).resolve()


def configured_model_files(config) -> List[Tuple[str, Path]]:
    """Return (role, resolved path) entries for configured model files."""
    config_path = str(config.config_path)
    entries = []
    main_path = str(config.get("model.path", "") or "").strip()
    if main_path:
        entries.append(("主模型", resolve_config_relative_path(main_path, config_path)))

    aux_path = str(config.get("model.aux_model_path", "") or "").strip()
    if aux_path:
        entries.append(("辅助模型", resolve_config_relative_path(aux_path, config_path)))
    return entries


def missing_model_files(config) -> List[Tuple[str, Path]]:
    """Return configured model files that do not exist."""
    return [(role, path) for role, path in configured_model_files(config)
            if not path.is_file()]


def validate_model_files(config) -> None:
    """Raise with a complete list when any configured model file is missing."""
    missing = missing_model_files(config)
    if not missing:
        return
    details = "；".join(f"{role}: {path}" for role, path in missing)
    raise ModelValidationError(f"配置的模型文件不存在：{details}")


def model_paths_for_report(config) -> Iterable[Tuple[str, Path]]:
    """Expose resolved paths for startup and evaluation reports."""
    return configured_model_files(config)
