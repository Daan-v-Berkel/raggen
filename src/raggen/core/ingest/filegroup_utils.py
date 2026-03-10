from __future__ import annotations

from typing import Dict, Tuple

from raggen.core.config.project import (
    ProjectConfig,
    GroupChunkingConfig,
)


class ConfigValidationError(ValueError):
    pass


def normalize_extension(ext: str) -> str:
    """
    Normalize a file extension for config/runtime lookup.

    Rules:
    - strip surrounding whitespace
    - lowercase
    - ensure leading '.'
    - reject empty values
    """
    ext = ext.strip().lower()
    if not ext:
        raise ConfigValidationError("Empty file extension is not allowed.")

    if not ext.startswith("."):
        ext = f".{ext}"

    if ext == ".":
        raise ConfigValidationError("Invalid file extension '.'.")
    return ext


def validate_file_groups(cfg: ProjectConfig) -> None:
    """
    Validate file group configuration.

    Checks:
    - fallback_group exists in file_groups
    - fallback_group exists in chunking
    - every file group has matching chunking config
    - every chunking config refers to a known file group
    - extensions are normalized and unique across groups
    """
    if cfg.fallback_group not in cfg.file_groups:
        raise ConfigValidationError(
            f"fallback_group '{cfg.fallback_group}' is missing from file_groups."
        )

    if cfg.fallback_group not in cfg.chunking:
        raise ConfigValidationError(
            f"fallback_group '{cfg.fallback_group}' is missing from chunking config."
        )

    for group_name in cfg.file_groups:
        if group_name not in cfg.chunking:
            raise ConfigValidationError(
                f"Missing chunking config for file group '{group_name}'."
            )

    for group_name in cfg.chunking:
        if group_name not in cfg.file_groups:
            raise ConfigValidationError(
                f"Chunking config exists for unknown file group '{group_name}'."
            )

    # Validate extension uniqueness and normalization
    seen: dict[str, str] = {}
    for group_name, group_cfg in cfg.file_groups.items():
        for raw_ext in group_cfg.extensions:
            ext = normalize_extension(raw_ext)

            if ext in seen:
                raise ConfigValidationError(
                    f"Extension '{ext}' is assigned to multiple groups: "
                    f"'{seen[ext]}' and '{group_name}'."
                )

            seen[ext] = group_name


def build_extension_group_map(cfg: ProjectConfig) -> Dict[str, str]:
    """
    Build extension -> group lookup.

    Assumes config has already been validated.
    """
    ext_to_group: dict[str, str] = {}

    for group_name, group_cfg in cfg.file_groups.items():
        for raw_ext in group_cfg.extensions:
            ext = normalize_extension(raw_ext)
            ext_to_group[ext] = group_name

    return ext_to_group


def build_group_chunking_map(cfg: ProjectConfig) -> Dict[str, GroupChunkingConfig]:
    """
    Build group -> chunking config lookup.

    Assumes config has already been validated.
    """
    return dict(cfg.chunking)


def build_filegroup_runtime_maps(
    cfg: ProjectConfig,
) -> Tuple[Dict[str, str], Dict[str, GroupChunkingConfig]]:
    """
    Validate config and derive both runtime lookup maps:

    - extension -> group
    - group -> chunking config
    """
    validate_file_groups(cfg)
    ext_to_group = build_extension_group_map(cfg)
    group_to_chunking = build_group_chunking_map(cfg)
    return ext_to_group, group_to_chunking
