import pytest

from raggen.core.config.project import (
    ProjectConfig,
    FileGroupConfig,
    GroupChunkingConfig,
)
from raggen.core.config.config_utils import (
    validate_file_groups,
    ConfigValidationError,
)


def make_cfg() -> ProjectConfig:
    return ProjectConfig(
        fallback_group="fallback",
        file_groups={
            "fallback": FileGroupConfig(extensions=[]),
            "documents": FileGroupConfig(extensions=[".md", ".txt"]),
        },
        chunking={
            "fallback": GroupChunkingConfig(
                strategy="fixed",
                chunk_size=1000,
                overlap=100,
            ),
            "documents": GroupChunkingConfig(
                strategy="headingAware",
                chunk_size=1200,
                overlap=100,
            ),
        },
    )


def test_validate_file_groups_accepts_valid_config():
    cfg = make_cfg()
    validate_file_groups(cfg)


def test_validate_file_groups_requires_fallback_group_in_file_groups():
    cfg = make_cfg()
    del cfg.file_groups["fallback"]

    with pytest.raises(ConfigValidationError, match="fallback_group"):
        validate_file_groups(cfg)


def test_validate_file_groups_requires_fallback_group_in_chunking():
    cfg = make_cfg()
    del cfg.chunking["fallback"]

    with pytest.raises(ConfigValidationError, match="fallback_group"):
        validate_file_groups(cfg)


def test_validate_file_groups_requires_matching_chunking_config():
    cfg = make_cfg()
    cfg.file_groups["python"] = FileGroupConfig(extensions=[".py"])

    with pytest.raises(ConfigValidationError, match="Missing chunking config"):
        validate_file_groups(cfg)


def test_validate_file_groups_rejects_orphan_chunking_config():
    cfg = make_cfg()
    cfg.chunking["python"] = GroupChunkingConfig(
        strategy="fixed",
        chunk_size=800,
        overlap=50,
    )

    with pytest.raises(ConfigValidationError, match="unknown file group"):
        validate_file_groups(cfg)


def test_validate_file_groups_rejects_duplicate_extensions():
    cfg = make_cfg()
    cfg.file_groups["notes"] = FileGroupConfig(extensions=[".md"])
    cfg.chunking["notes"] = GroupChunkingConfig(
        strategy="fixed",
        chunk_size=1000,
        overlap=100,
    ),

    with pytest.raises(ConfigValidationError, match=r".md"):
        validate_file_groups(cfg)


def test_validate_file_groups_normalizes_extensions():
    cfg = make_cfg()
    cfg.file_groups["python"] = FileGroupConfig(extensions=["PY"])
    cfg.chunking["python"] = GroupChunkingConfig(
        strategy="fixed",
        chunk_size=800,
        overlap=50,
    )

    # should not fail just because extension is missing "."
    validate_file_groups(cfg)
