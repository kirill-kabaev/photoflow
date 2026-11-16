import json

import numpy as np
import pytest

from photoflow import imaging, queue
from photoflow.models import Plan
from photoflow.storage import Session, digest, write_json
from photoflow.workflow import Workflow


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "shoot"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "frame.ARW").write_bytes(b"synthetic source, NOT a decodable RAW")
    return Session(root)


@pytest.fixture
def pixels():
    y, x = np.mgrid[:80, :120]
    return np.stack([0.1 + x / 400, 0.1 + y / 400, np.full_like(x, 0.12, dtype=float)], axis=-1).astype(np.float32)


@pytest.fixture
def decoder(pixels):
    def decode(path, settings):
        # A deliberate synthetic decoder: ordinary tests never claim real RAW decoding.
        return imaging.Decoded(pixels.copy(), {"oriented_dimensions": [120, 80],
                            "synthetic_decoder": True, "raw_settings": settings.model_dump()})
    return decode


@pytest.fixture
def workflow(session, decoder):
    queue.scan(session)
    return Workflow(session, decoder=decoder)


def make_plan(workflow, photo, parent=None, exposure=0.25, feedback=None):
    context = json.loads(workflow.session.work(photo.id, "context.json").read_text())
    payload = {
        "analysis": {"photo_id": photo.id, "source_sha256": photo.source_sha256,
                     "preview_sha256": context["preview_sha256"], "provider": "test_fixture",
                     "model": "synthetic-test-only", "visually_inspected": True,
                     "category": "other", "people_present": False,
                     "observations": ["Synthetic response; no real visual AI ran"],
                     "technical": {"exposure": "Synthetic test judgment"},
                     "recommendations": ["Exercise deterministic exposure"],
                     "uncertainties": ["Not a real AI analysis"]},
        "recipe": {"photo_id": photo.id, "source_sha256": photo.source_sha256,
                   "parent_version": parent, "style": "synthetic_test",
                   "rationale": "Synthetic recipe for workflow verification",
                   "global_adjustments": {"exposure_ev": exposure}},
    }
    if feedback:
        payload["recipe"].update(feedback_applied="Synthetic fixture incorporates requested exposure change",
                                 feedback_sha256=digest(feedback))
    plan = Plan.model_validate_json(json.dumps(payload))
    path = workflow.session.work(photo.id, "test-plan.json")
    write_json(path, plan)
    return path


def process_first(workflow):
    photo = workflow.queue.photos[0]
    workflow.prepare(photo.id)
    path = make_plan(workflow, photo)
    workflow.import_plan(photo.id, path)
    workflow.process(photo.id)
    return photo


def import_review(workflow, photo, recommendation="human_review"):
    context = workflow.review(photo.id)
    payload = {"photo_id": photo.id, "version": photo.current_version,
               "comparison_sha256": context["comparison_sha256"],
               "provider": "test_fixture", "model": "synthetic-test-only", "visually_inspected": True,
               "improvements": ["Synthetic review"], "concerns": [],
               "recommendation": recommendation, "rationale": "Workflow test; no real AI review"}
    path = workflow.session.work(photo.id, "test-review.json")
    write_json(path, payload)
    workflow.review(photo.id, path)
    return path
