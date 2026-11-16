"""Atomic JSON persistence and advisory single-session locking for Linux/WSL2."""
import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class FlowError(Exception):
    """Expected operational error, suitable for CLI display."""


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_bytes(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes):
    if path.is_symlink():
        raise FlowError(f"Refusing symlink output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json_bytes(value))


def read_model(path, model):
    try:
        return model.model_validate_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise FlowError(f"Missing file: {path}") from exc


def checked_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", value):
        raise FlowError("Invalid photo identifier")
    return value


def checked_version(value):
    if not re.fullmatch(r"v[0-9]{3,}", value):
        raise FlowError("Invalid version identifier")
    return value


class Session:
    def __init__(self, root):
        original = Path(root).expanduser().absolute()
        if ".." in original.parts or any(p.is_symlink() for p in [original, *original.parents]):
            raise FlowError("Session path must not contain traversal or symlinks")
        if not original.is_dir():
            raise FlowError(f"Session does not exist: {original}; create SESSION/raw first")
        self.root = original.resolve()
        raw = self.path("raw")
        if not raw.is_dir():
            raise FlowError("Session must have an existing raw/ directory")

    def path(self, *parts):
        target = self.root
        for part in parts:
            piece = Path(part)
            if piece.is_absolute() or not piece.parts or any(p in {"..", "."} for p in piece.parts):
                raise FlowError("Unsafe relative path")
            for component in piece.parts:
                target /= component
                if target.is_symlink():
                    raise FlowError(f"Symlinks are unsupported: {target}")
        if not target.resolve().is_relative_to(self.root):
            raise FlowError("Path escapes session")
        return target

    def source(self, photo):
        if Path(photo.filename).name != photo.filename or photo.filename in {".", ".."}:
            raise FlowError("Invalid source filename in queue")
        path = self.path("raw", photo.filename)
        if not path.is_file():
            raise FlowError(f"Source missing: {path}")
        if digest(path) != photo.source_sha256:
            raise FlowError(f"Source checksum changed: {photo.filename}; original is not overwritten")
        return path

    def work(self, photo_id, *parts):
        return self.path("work", checked_id(photo_id), *parts)

    @contextmanager
    def lock(self):
        path = self.path(".photoflow.lock")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FlowError("Session is busy; another PhotoFlow operation holds its lock") from exc
            yield self
        finally:
            os.close(fd)
