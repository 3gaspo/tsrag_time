"""Stage a complete report bundle before replacing its published directory."""

from __future__ import annotations

import atexit
import os
import shutil
import uuid
from pathlib import Path


class ReportTransaction:
    """Keep the previous report visible until the replacement is complete."""

    def __init__(self, destination: str | Path):
        self.destination = Path(destination).expanduser().resolve()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{self.destination.name}.staging-"
        self.staging = self.destination.parent / f"{prefix}{uuid.uuid4().hex}"
        self.staging.mkdir()
        self._committed = False
        atexit.register(self.abort)

    def abort(self) -> None:
        """Remove only this process's unpublished staging directory."""
        if self.staging.exists():
            shutil.rmtree(self.staging)

    def path(self, destination_path: str | Path) -> Path:
        path = Path(destination_path).expanduser().resolve()
        try:
            relative = path.relative_to(self.destination)
        except ValueError as error:
            raise ValueError(
                f"Report artifact {path} is outside {self.destination}"
            ) from error
        return self.staging / relative

    def commit(self) -> Path:
        if self._committed:
            return self.destination
        backup = self.destination.parent / (
            f".{self.destination.name}.previous-{uuid.uuid4().hex}"
        )
        had_previous = self.destination.exists()
        if had_previous:
            os.replace(self.destination, backup)
        try:
            os.replace(self.staging, self.destination)
        except BaseException:
            if had_previous and backup.exists() and not self.destination.exists():
                os.replace(backup, self.destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        self._committed = True
        return self.destination
