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
      - ** prefix: **/pattern matches at any depth including root

    Notes:
      - Intentionally lean, not a full re-implementation of git's ignore engine.
      - Rules are evaluated in order; the last matching rule wins.
    """

    def __init__(self, rules: Iterable[_Rule]) -> None:
        self._rules = list(rules)

    @staticmethod
    def _parse_lines(lines: Iterable[str]) -> list[_Rule]:
        rules: list[_Rule] = []
        for raw in lines:
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

            rules.append(
                _Rule(
                    pattern=line.replace(os.sep, "/"),
                    negated=negated,
                    dir_only=dir_only,
                    anchored=anchored,
                )
            )
        return rules

    @staticmethod
    def from_ignore_files(root: Path, ignore_files: List[str | Path]) -> "GitIgnoreLike":
        rules: list[_Rule] = []
        for file in ignore_files:
            ignore_file = root / file
            if not ignore_file.exists():
                continue
            lines = ignore_file.read_text(encoding="utf-8", errors="replace").splitlines()
            rules.extend(GitIgnoreLike._parse_lines(lines))
        return GitIgnoreLike(rules)

    @staticmethod
    def from_patterns(patterns: List[str]) -> "GitIgnoreLike":
        """Build a matcher from a list of raw inline pattern strings."""
        return GitIgnoreLike(GitIgnoreLike._parse_lines(patterns))

    def check(self, rel_posix_path: str, is_dir: bool) -> bool | None:
        """
        Returns True if ignored, False if explicitly un-ignored, or None if no
        rule matched. Callers that combine multiple scoped matchers use None to
        distinguish "no opinion" from "explicitly not ignored".
        """
        rel_posix_path = rel_posix_path.lstrip("/")
        last_match: bool | None = None
        for rule in self._rules:
            if rule.dir_only and not is_dir:
                continue
            if self._match(rule, rel_posix_path, is_dir=is_dir):
                last_match = not rule.negated
        return last_match

    def ignores(self, rel_posix_path: str, is_dir: bool) -> bool:
        """Returns True if the path is ignored, False otherwise."""
        return self.check(rel_posix_path, is_dir) is True

    def _match(self, rule: _Rule, rel_path: str, is_dir: bool) -> bool:
        # For directories, also check the path with a trailing slash so that
        # dir-only patterns without an explicit slash still match.
        candidates = [rel_path]
        if is_dir and not rel_path.endswith("/"):
            candidates.append(rel_path + "/")

        if rule.anchored:
            # Anchored patterns match from the repo root only.
            return any(self._fnmatch_posix(c, rule.pattern) for c in candidates)

        if "/" in rule.pattern:
            # A slash in the pattern means it must be matched against the full
            # relative path, optionally prefixed with **/ for unanchored use.
            return any(
                self._fnmatch_posix(c, f"**/{rule.pattern}")
                or self._fnmatch_posix(c, rule.pattern)
                for c in candidates
            )

        # No slash: match if the pattern matches any single path segment.
        parts = rel_path.split("/")
        return any(fnmatch.fnmatchcase(p, rule.pattern) for p in parts)

    @staticmethod
    def _fnmatch_posix(path: str, pat: str) -> bool:
        if fnmatch.fnmatchcase(path, pat) or fnmatch.fnmatchcase(path, pat.rstrip("/")):
            return True
        # **/foo must also match foo at the root level (zero directory depth).
        if pat.startswith("**/"):
            suffix = pat[3:]
            return (
                fnmatch.fnmatchcase(path, suffix)
                or fnmatch.fnmatchcase(path, suffix.rstrip("/"))
            )
        return False


def _is_ignored(
    scoped_matchers: list[tuple[str, GitIgnoreLike]],
    rel_path: str,
    is_dir: bool,
) -> bool:
    """
    Evaluate scoped matchers in order (root first, deepest last).
    The last rule that has any opinion across all applicable scopes wins.
    A scoped matcher at prefix 'p' only applies to paths under 'p/'.
    """
    last_match: bool | None = None
    for scope, matcher in scoped_matchers:
        if scope and not rel_path.startswith(scope + "/"):
            continue
        local = rel_path[len(scope) + 1:] if scope else rel_path
        result = matcher.check(local, is_dir)
        if result is not None:
            last_match = result
    return last_match is True


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
    ignore_patterns: List[str] | None = None,
    follow_symlinks: bool = False,
    include_hidden: bool = True,
) -> ScanResult:
    """
    Recursively scan from root_dir, respecting ignore rules from:
      - <root_dir>/<ignore_filename> and any nested copies in subdirectories
      - inline ignore_patterns (e.g. from scan.ignore config)

    Always ignores the .rag/ directory. Rules use last-match-wins semantics
    across all applicable ignore files, mirroring git's behaviour.
    """
    root = Path(root_dir).resolve()

    root_matcher = GitIgnoreLike(
        GitIgnoreLike.from_ignore_files(root, ignore_filenames)._rules
        + GitIgnoreLike.from_patterns(ignore_patterns or [])._rules
    )
    # scoped_matchers: list of (scope_prefix, matcher)
    # scope "" means rules apply from the project root.
    scoped_matchers: list[tuple[str, GitIgnoreLike]] = [("", root_matcher)]

    cfg = ProjectConfig.get_config()
    groups = {n: [] for n in cfg.file_groups.keys()}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dpath = Path(dirpath)
        rel_dir = dpath.relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # Load ignore files from this subdirectory (root is already loaded above).
        if rel_dir:
            sub_matcher = GitIgnoreLike.from_ignore_files(dpath, ignore_filenames)
            if sub_matcher._rules:
                scoped_matchers.append((rel_dir, sub_matcher))

        # Prune directories before descending.
        kept_dirnames: list[str] = []
        for dn in dirnames:
            if dn == ".rag":
                continue
            if not include_hidden and dn.startswith("."):
                continue
            rel = f"{rel_dir}/{dn}" if rel_dir else dn
            if _is_ignored(scoped_matchers, rel, is_dir=True):
                continue
            kept_dirnames.append(dn)
        dirnames[:] = kept_dirnames

        for fn in filenames:
            if not include_hidden and fn.startswith("."):
                continue
            p = dpath / fn

            try:
                st = p.stat()
            except FileNotFoundError:
                continue

            if not os.path.isfile(p):
                continue

            rel = (p.relative_to(root)).as_posix()
            if _is_ignored(scoped_matchers, rel, is_dir=False):
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
