"""Per-run manifests for auditability.

`RunManifest` is a context manager wrapped around each CLI subcommand. It records
what ran (stage, argv), the code version (git SHA, package version), timing, and
the outcome (success/failure + exception), then writes a JSON record under
``<data_root>/_runs/``. Output artifacts discovered after the run can be attached
with ``record_outputs`` so each manifest also captures what it produced.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("swp-data")
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _count_outputs(paths: list[Path]) -> dict:
    """Summarize output artifacts: existing paths and total file count."""
    summary = []
    for p in paths:
        if not p.exists():
            summary.append({"path": str(p), "exists": False, "n_files": 0})
        elif p.is_dir():
            n = sum(1 for f in p.rglob("*") if f.is_file())
            summary.append({"path": str(p), "exists": True, "n_files": n})
        else:
            summary.append({"path": str(p), "exists": True, "n_files": 1})
    return {"outputs": summary}


class RunManifest(AbstractContextManager):
    """Record one pipeline-stage invocation to ``<runs_dir>/{ts}_{stage}.json``."""

    def __init__(self, stage: str, runs_dir: Path, args: dict | None = None) -> None:
        self.stage = stage
        self.runs_dir = Path(runs_dir)
        self.record: dict = {
            "stage": stage,
            "args": args or {},
            "argv": sys.argv,
            "git_sha": _git_sha(),
            "package_version": _package_version(),
            "python": platform.python_version(),
            "started_at": None,
            "ended_at": None,
            "duration_s": None,
            "status": "running",
            "error": None,
            "outputs": [],
        }
        self._t0 = 0.0
        self._started = datetime.now(timezone.utc)

    def record_outputs(self, paths: list[Path]) -> None:
        self.record["outputs"] = _count_outputs(paths)["outputs"]

    def __enter__(self) -> "RunManifest":
        self._t0 = time.monotonic()
        self.record["started_at"] = self._started.isoformat()
        logger.info("stage '%s' started", self.stage)
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> bool:
        self.record["duration_s"] = round(time.monotonic() - self._t0, 3)
        self.record["ended_at"] = datetime.now(timezone.utc).isoformat()
        if exc is None:
            self.record["status"] = "success"
            logger.info("stage '%s' succeeded in %.1fs",
                        self.stage, self.record["duration_s"])
        else:
            self.record["status"] = "failed"
            self.record["error"] = f"{exc_type.__name__}: {exc}"
            logger.error("stage '%s' failed after %.1fs: %s",
                         self.stage, self.record["duration_s"], self.record["error"])
        self._write()
        return False  # never suppress the exception

    def _write(self) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            ts = self._started.strftime("%Y%m%dT%H%M%SZ")
            dest = self.runs_dir / f"{ts}_{self.stage}.json"
            dest.write_text(json.dumps(self.record, indent=2, default=str))
            logger.debug("wrote run manifest %s", dest)
        except OSError as err:
            logger.warning("could not write run manifest: %s", err)
