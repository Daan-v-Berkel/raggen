from __future__ import annotations

import fnmatch
import hashlib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from raggen.core.config.project import ProjectConfig


@dataclass(frozen=True)
class FileRef:
    path: str  # absolute path
    relative_path: str  # relative to root, posix-style
    file_size: int
    mtime: int
    content_hash: str  # sha256 hex digest
    mime_type: str


@dataclass
class ScanResult:
    groups: dict[str, list[FileRef]]


@dataclass(frozen=True)
class _Rule:
    pattern: str
    negated: bool
    dir_only: bool
    anchored: bool


class GitIgnoreLike:
    """
    Minimal .gitignore-style matcher.

    Supported:
      - comments (#...), blank lines
      - negation: !pattern
      - anchored patterns: /foo/bar
      - directory-only: pattern ending with /
      - wildcards via fnmatch (*, ?, [abc])
      - ** (treated leniently; works well enough for common cases)

    Notes:
      - This is intentionally lean, not a full re-implementation of git's ignore engine.
      - Rules are evaluated in order; the last matching rule wins.
    """

    def __init__(self, rules: Iterable[_Rule]) -> None:
        self._rules = list(rules)

    @staticmethod
    def from_ignore_files(root, ignore_files: List[Path]) -> "GitIgnoreLike":
        rules: list[_Rule] = []
        for file in ignore_files:
            ignore_file = root / file
            if not ignore_file.exists():
                return GitIgnoreLike(rules)

            for raw in ignore_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                line = raw.strip("\n\r")
                if not line or line.lstrip().startswith("#"):
                    continue

                # allow escaping a leading '#' or '!' with backslash
                if line.startswith(r"\#") or line.startswith(r"\!"):
                    line = line[1:]

                negated = line.startswith("!")
                if negated:
                    line = line[1:]

                anchored = line.startswith("/")
                if anchored:
                    line = line[1:]

                dir_only = line.endswith("/")
                if dir_only:
                    line = line[:-1]

                line = line.strip()
                if not line:
                    continue

                # normalize to posix-ish matching
                pattern = line.replace(os.sep, "/")
                rules.append(
                    _Rule(
                        pattern=pattern,
                        negated=negated,
                        dir_only=dir_only,
                        anchored=anchored,
                    )
                )

        return GitIgnoreLike(rules)

    def ignores(self, rel_posix_path: str, is_dir: bool) -> bool:
        """
        rel_posix_path: path relative to root, using forward slashes.
        """
        ignored = False

        # Git matches paths without leading slash.
        rel_posix_path = rel_posix_path.lstrip("/")

        for rule in self._rules:
            if rule.dir_only and not is_dir:
                continue

            if self._match(rule, rel_posix_path, is_dir=is_dir):
                ignored = not rule.negated

        return ignored

    def _match(self, rule: _Rule, rel_path: str, is_dir: bool) -> bool:
        # Directory paths: git effectively matches "dir" and "dir/**"
        # We'll check both forms to cover common cases.
        candidates = [rel_path]
        if is_dir and not rel_path.endswith("/"):
            candidates.append(rel_path + "/")

        if rule.anchored:
            # anchored means match from repo root
            return any(self._fnmatch_posix(c, rule.pattern) for c in candidates)

        # unanchored rules can match any path segment; easiest approximation:
        # - if pattern contains '/', match against full rel path (anywhere)
        # - else match against basename and any segment
        if "/" in rule.pattern:
            return any(
                self._fnmatch_posix(c, f"**/{rule.pattern}")
                or self._fnmatch_posix(c, rule.pattern)
                for c in candidates
            )

        # no slash: match basename OR any segment
        parts = rel_path.split("/")
        return any(
            fnmatch.fnmatchcase(p, rule.pattern) for p in parts
        ) or fnmatch.fnmatchcase(parts[-1], rule.pattern)

    @staticmethod
    def _fnmatch_posix(path: str, pat: str) -> bool:
        # A small helper to make ** behave reasonably with fnmatch.
        # fnmatch supports '*' matching slashes on most platforms in Python's implementation.
        # We just use fnmatchcase on posix strings.
        return fnmatch.fnmatchcase(path, pat) or fnmatch.fnmatchcase(
            path, pat.rstrip("/")
        )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _guess_mime_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def _resolve_group(ext: str) -> str:
    cfg = ProjectConfig.get_config()
    cfg_groups = cfg.file_groups
    for groupname, fg in cfg_groups.items():
        if ext in fg.extensions:
            return groupname
    return cfg.fallback_group


def scan_files(
    root_dir: str | Path,
    *,
    ignore_filenames: List[str],
    follow_symlinks: bool = False,
    include_hidden: bool = True,
) -> ScanResult:
    """
    Recursively scan from root_dir and yield FileRef for each file encountered,
    respecting ignore rules from <root_dir>/<ignore_filename>.
    And always ingoring it's own directory (.rag/).

    ignore rules apply to *relative paths* under root_dir.
    """
    root = Path(root_dir).resolve()
    ignore = GitIgnoreLike.from_ignore_files(root, ignore_filenames)
    # groups: dict[str, list[FileRef]] = {
    #     "code": [],
    #     "document": [],
    #     "fallback": [],
    # }

    cfg = ProjectConfig.get_config()
    groups = {n: [] for n in cfg.file_groups.keys()}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dpath = Path(dirpath)
        rel_dir = dpath.relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # prune dirs based on ignore rules
        kept_dirnames: list[str] = []
        for dn in dirnames:
            if dn == ".rag":
                continue
            if not include_hidden and dn.startswith("."):
                continue
            rel = f"{rel_dir}/{dn}" if rel_dir else dn
            if ignore.ignores(rel, is_dir=True):
                continue
            kept_dirnames.append(dn)
        dirnames[:] = kept_dirnames

        # files
        for fn in filenames:
            if not include_hidden and fn.startswith("."):
                continue
            p = dpath / fn

            # skip non-regular files
            try:
                st = p.stat()
            except FileNotFoundError:
                continue

            if not os.path.isfile(p):
                continue

            rel = (p.relative_to(root)).as_posix()
            if ignore.ignores(rel, is_dir=False):
                continue

            ext = Path(rel).suffix.lower()
            group = _resolve_group(ext)

            fr = FileRef(
                path=str(p),
                relative_path=rel,
                file_size=int(st.st_size),
                mtime=int(st.st_mtime),
                content_hash=_sha256_file(p),
                mime_type=_guess_mime_type(p),
            )

            groups[group].append(fr)

    return ScanResult(groups=groups)
