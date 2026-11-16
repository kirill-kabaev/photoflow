import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from photoflow import queue
from photoflow.models import Queue
from photoflow.storage import FlowError, Session, read_model


def test_scan_stable_collision_safe_and_preserves_state(session):
    raw = session.path("raw")
    (raw / "frame.NEF").write_bytes(b"different format")
    (raw / "B.ARW").write_bytes(b"B")
    (raw / "a.ARW").write_bytes(b"a")
    first = queue.scan(session)
    assert [p.filename for p in first.photos] == ["a.ARW", "B.ARW", "frame.ARW", "frame.NEF"]
    assert len({p.id for p in first.photos}) == 4
    queue.transition(first.photos[0], "analyzing", "test")
    queue.save(session, first)
    second = queue.scan(session)
    assert len(second.photos) == 4
    assert second.photos[0].state == "analyzing"
    with pytest.raises(FlowError, match="ambiguous"):
        queue.find(second, "frame")


def test_changed_raw_rejected_without_queue_loss(session):
    queue.scan(session)
    snapshot = session.path("queue.json").read_bytes()
    session.path("raw", "frame.ARW").write_bytes(b"user replaced original")
    with pytest.raises(FlowError, match="changed"):
        queue.scan(session)
    assert session.path("queue.json").read_bytes() == snapshot


def test_missing_raw_not_silently_skipped(session):
    queue.scan(session)
    session.path("raw", "frame.ARW").unlink()
    with pytest.raises(FlowError, match="missing"):
        queue.scan(session)


def test_transitions_priority_and_queue_validation(session):
    q = queue.scan(session)
    p = q.photos[0]
    with pytest.raises(FlowError, match="transition"):
        queue.transition(p, "approved", "invalid")
    queue.set_priority(p, 50, "Representative frame")
    assert queue.next_photo(q).id == p.id
    assert p.history[-1]["priority_before"] == 0
    p.state = "approved"
    with pytest.raises(ValidationError, match="provenance"):
        queue.save(session, q)


def test_lock_blocks_another_process(session):
    code = "from photoflow.storage import Session; import sys; s=Session(sys.argv[1]);\nwith s.lock(): print('acquired')"
    with session.lock():
        result = subprocess.run([sys.executable, "-c", code, str(session.root)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Session is busy" in result.stderr
    with session.lock():
        pass


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "work/../../outside"])
def test_traversal_rejected(session, relative):
    with pytest.raises(FlowError):
        session.path(relative)


def test_symlink_sources_and_artifact_directories_rejected(session, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    session.path("raw", "alias.ARW").symlink_to(session.path("raw", "frame.ARW"))
    with pytest.raises(FlowError, match="symlink"):
        queue.scan(session)
    session.path("work").symlink_to(outside, target_is_directory=True)
    with pytest.raises(FlowError, match="Symlink"):
        session.work("frame")


def test_corrupt_queue_and_duplicate_entries_fail(session):
    q = queue.scan(session)
    q.photos.append(q.photos[0].model_copy())
    with pytest.raises(ValidationError, match="duplicate"):
        queue.save(session, q)
    session.path("queue.json").write_text('{"incomplete":')
    with pytest.raises(ValidationError):
        queue.scan(session)


def test_atomic_write_failure_preserves_previous_json(session, monkeypatch):
    import photoflow.storage as storage
    storage.write_json(session.path("state.json"), {"value": "old"})
    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError):
        storage.write_json(session.path("state.json"), {"value": "new"})
    assert json.loads(session.path("state.json").read_text()) == {"value": "old"}
    assert list(session.root.glob(".tmp-*")) == []


def test_identical_content_registered_as_visible_duplicate(session):
    session.path("raw", "copy.ARW").write_bytes(session.path("raw", "frame.ARW").read_bytes())
    q = queue.scan(session)
    assert len(q.photos) == 2
    representative = next(p for p in q.photos if p.duplicate_of is None)
    duplicate = next(p for p in q.photos if p.duplicate_of is not None)
    assert duplicate.duplicate_of == representative.id
    assert queue.next_photo(q).id == representative.id
    # Direct selection remains an explicit supported choice.
    assert queue.find(q, duplicate.id) == duplicate
