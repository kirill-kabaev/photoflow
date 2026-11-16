"""Explicit queue transitions; no AI judgments are made here."""
import hashlib
import re
from pathlib import Path

from .models import Photo, Queue, SessionStyle
from .storage import FlowError, checked_id, digest, now, read_model, write_json

RAW_EXTENSIONS = {".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2", ".orf", ".pef", ".srw"}
TRANSITIONS = {
    "pending": {"analyzing", "failed"},
    "analyzing": {"ready", "failed"},
    "ready": {"processing", "analyzing", "revision_requested", "failed"},
    "processing": {"review", "failed"},
    "review": {"revision_requested", "approved", "failed"},
    "revision_requested": {"ready", "failed"},
    "approved": {"exporting", "revision_requested", "failed"},
    "exporting": {"exported", "failed"},
    "exported": {"revision_requested", "failed"},
    "failed": {"analyzing", "revision_requested", "review", "approved"},
}


def load(session):
    return read_model(session.path("queue.json"), Queue)


def save(session, queue):
    queue.updated_at = now()
    # Revalidate mutations as well as deserialized data.
    validated = Queue.model_validate_json(queue.model_dump_json())
    write_json(session.path("queue.json"), validated)


def transition(photo, target, reason):
    if target not in TRANSITIONS[photo.state]:
        raise FlowError(f"Invalid state transition {photo.state} -> {target}")
    photo.history.append({"at": now(), "from": photo.state, "to": target, "reason": reason})
    photo.state, photo.updated_at = target, now()
    if target != "failed":
        photo.error = None


def find(queue, identifier):
    checked_id(identifier)
    exact = [p for p in queue.photos if p.id == identifier]
    matches = exact or [p for p in queue.photos if Path(p.filename).stem == identifier]
    if len(matches) != 1:
        raise FlowError("Photo not found or ambiguous; use ID from photoflow queue")
    return matches[0]


def stable_id(filename):
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:60].strip("_-") or "photo"
    return f"{stem}-{hashlib.sha256(filename.encode()).hexdigest()[:12]}"


def scan(session):
    queue_path = session.path("queue.json")
    queue = load(session) if queue_path.exists() else Queue(updated_at=now())
    known = {p.filename: p for p in queue.photos}
    checksums = {p.source_sha256: p.id for p in queue.photos if not p.duplicate_of}
    files = sorted(session.path("raw").iterdir(), key=lambda p: (p.name.casefold(), p.name))
    for path in files:
        if path.suffix.lower() not in RAW_EXTENSIONS:
            continue
        if path.is_symlink() or not path.is_file():
            raise FlowError(f"RAW must be a regular file, not a symlink: {path.name}")
        checksum = digest(path)
        if path.name in known:
            if checksum != known[path.name].source_sha256:
                raise FlowError(f"Registered RAW changed: {path.name}")
            continue
        photo_id = stable_id(path.name)
        queue.photos.append(Photo(id=photo_id, filename=path.name,
                                  source_sha256=checksum, duplicate_of=checksums.get(checksum), updated_at=now()))
        checksums.setdefault(checksum, photo_id)
    missing = [p.filename for p in queue.photos if not session.path("raw", p.filename).is_file()]
    if missing:
        raise FlowError(f"Registered sources missing: {missing}")
    queue.photos.sort(key=lambda p: (p.filename.casefold(), p.filename))
    session.path("work").mkdir(exist_ok=True)
    session.path("final").mkdir(exist_ok=True)
    style = session.path("session_style.json")
    if not style.exists():
        write_json(style, SessionStyle())
    else:
        read_model(style, SessionStyle)
    save(session, queue)
    return queue


def next_photo(queue):
    # Revisions and unfinished analysis must not silently be skipped for new work.
    actionable = {"revision_requested", "ready", "analyzing", "pending"}
    ordered = sorted(enumerate(queue.photos), key=lambda pair: (-pair[1].priority, pair[0]))
    return next((p for _, p in ordered if p.state in actionable and not p.duplicate_of), None)


def set_priority(photo, priority, reason):
    if not reason.strip():
        raise FlowError("Priority change requires a reason")
    photo.history.append({"at": now(), "priority_before": photo.priority,
                          "priority_after": priority, "reason": reason})
    photo.priority = priority
    photo.updated_at = now()
