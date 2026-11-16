import json
import re
from pathlib import Path

import pytest
from PIL import Image

from conftest import import_review, make_plan, process_first
from photoflow import queue, reporting
from photoflow.cli import main
from photoflow.storage import FlowError, digest, write_json
from photoflow.workflow import Workflow


def test_end_to_end_revision_approval_export(workflow):
    """Real processing + file lifecycle; decoder and AI are explicit synthetic fixtures."""
    session = workflow.session
    source = session.path("raw", "frame.ARW")
    original = digest(source)
    photo = process_first(workflow)
    assert photo.state == "review" and photo.current_version == "v001"
    first = session.work(photo.id, "v001")
    old_recipe = (first / "recipe.json").read_bytes()
    feedback = first / "feedback.md"
    feedback.write_text("Сделать светлее на 0.1 EV, сохранить цвет и текстуру.\n")
    review = workflow.request_revision(photo.id)
    assert review["feedback"] == feedback.read_text()
    plan = make_plan(workflow, photo, parent="v001", exposure=0.35, feedback=feedback)
    workflow.import_plan(photo.id, plan)
    workflow.process(photo.id)
    second = session.work(photo.id, "v002")
    assert photo.versions == ["v001", "v002"]
    assert (first / "recipe.json").read_bytes() == old_recipe
    assert (second / "feedback_applied.md").read_text() == feedback.read_text()
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["parent_version"] == "v001"
    with pytest.raises(FlowError, match="approved"):
        workflow.export(photo.id)
    with pytest.raises(FlowError, match="review missing"):
        workflow.approve(photo.id, "v002", "Test user")
    import_review(workflow, photo)
    workflow.approve(photo.id, "v002", "Test user")
    result = workflow.export(photo.id)
    assert photo.state == "exported"
    assert result["validation"]["tiff_bits_per_channel"] == 16
    final_manifest = json.loads(session.path("final", f"{photo.id}.manifest.json").read_text())
    assert final_manifest["source_version"] == "v002"
    assert final_manifest["approved_by"] == "Test user"
    for name, checksum in final_manifest["files"].items():
        assert digest(session.path("final", name)) == checksum
    assert digest(source) == original
    assert (first / "recipe.json").read_bytes() == old_recipe
    for version in ["v001", "v002"]:
        folder = session.work(photo.id, version)
        for name in ["before.jpg", "after.jpg", "before_after.jpg", "report.md", "feedback.md", "recipe.json", "manifest.json", "ai_review.json"]:
            assert (folder / name).is_file()
        report = (folder / "report.md").read_text()
        for link in re.findall(r"\]\((\./[^)]+)\)", report):
            assert (folder / link).exists(), link
        assert "test_fixture" in report
    before_feedback = feedback.read_bytes()
    reporting.version_report(session, photo, "v001")
    assert feedback.read_bytes() == before_feedback
    overview = session.path("review.md").read_text()
    assert "exported" in overview and "v002/report.md" in overview


def test_reprocessing_requires_new_revision_and_existing_final_protected(workflow):
    photo = process_first(workflow)
    with pytest.raises(FlowError, match="ready"):
        workflow.process(photo.id)
    import_review(workflow, photo)
    workflow.approve(photo.id, "v001", "Test user")
    existing = workflow.session.path("final", f"{photo.id}.jpg")
    existing.write_bytes(b"previous valuable export")
    with pytest.raises(FlowError, match="refusing overwrite"):
        workflow.export(photo.id)
    assert existing.read_bytes() == b"previous valuable export"


def test_tampered_version_cannot_be_approved(workflow):
    photo = process_first(workflow)
    import_review(workflow, photo)
    folder = workflow.session.work(photo.id, "v001")
    recipe = json.loads((folder / "recipe.json").read_text())
    recipe["global_adjustments"]["exposure_ev"] = 2.0
    write_json(folder / "recipe.json", recipe)
    with pytest.raises(FlowError, match="artifact modified"):
        workflow.approve(photo.id, "v001", "Test user")


def test_plan_source_mismatch_and_stale_feedback_rejected(workflow):
    photo = workflow.queue.photos[0]
    workflow.prepare(photo.id)
    path = make_plan(workflow, photo)
    bad = json.loads(path.read_text())
    bad["analysis"]["source_sha256"] = bad["recipe"]["source_sha256"] = "0" * 64
    write_json(path, bad)
    with pytest.raises(FlowError, match="mismatch"):
        workflow.import_plan(photo.id, path)
    path = make_plan(workflow, photo)
    workflow.import_plan(photo.id, path)
    workflow.process(photo.id)
    feedback = workflow.session.work(photo.id, "v001", "feedback.md")
    feedback.write_text("Make brighter")
    workflow.request_revision(photo.id)
    path = make_plan(workflow, photo, parent="v001", feedback=feedback)
    feedback.write_text("Make darker instead")
    with pytest.raises(FlowError, match="checksum"):
        workflow.import_plan(photo.id, path)


def test_processing_failure_preserves_queue_and_retry_creates_new_version(workflow):
    photo = workflow.queue.photos[0]
    workflow.prepare(photo.id)
    workflow.import_plan(photo.id, make_plan(workflow, photo))
    decoder = workflow.decoder
    def fail(*args):
        raise FlowError("Simulated decoder failure")
    workflow.decoder = fail
    with pytest.raises(FlowError, match="Simulated"):
        workflow.process(photo.id)
    persisted = queue.load(workflow.session).photos[0]
    assert persisted.state == "failed"
    assert persisted.versions == []
    assert workflow.session.work(photo.id, ".staging-v001").is_dir()
    workflow.decoder = decoder
    workflow.prepare(photo.id)
    workflow.import_plan(photo.id, make_plan(workflow, workflow.photo(photo.id)))
    result = workflow.process(photo.id)
    assert result["version"] == "v002"  # Failed version numbers are never reused.


def test_recover_interrupted_processing_without_complete_version(workflow):
    photo = workflow.queue.photos[0]
    workflow.prepare(photo.id)
    workflow.import_plan(photo.id, make_plan(workflow, photo))
    queue.transition(photo, "processing", "simulated crash")
    photo.operation_version = "v001"
    workflow.session.work(photo.id, ".staging-v001").mkdir()
    workflow.save()
    recovered = Workflow(workflow.session, workflow.decoder).recover()
    assert recovered[0]["state"] == "failed"
    assert "Interrupted" in recovered[0]["error"]


def test_recover_complete_version_published_before_queue_save(workflow):
    photo = process_first(workflow)
    # Simulate an on-disk queue still in processing while v001 was fully published.
    q = queue.load(workflow.session)
    p = q.photos[0]
    p.state, p.current_version, p.versions, p.operation_version = "processing", None, [], "v001"
    queue.save(workflow.session, q)
    fresh = Workflow(workflow.session, workflow.decoder)
    result = fresh.recover()
    assert result[0]["state"] == "review"
    assert fresh.photo(photo.id).versions == ["v001"]


def test_real_analysis_is_not_claimed_by_prepare(workflow):
    photo = workflow.queue.photos[0]
    context = workflow.prepare(photo.id)
    assert context["ai_status"] == "awaiting_codex_visual_analysis"
    assert not workflow.session.work(photo.id, "analysis.json").exists()
    assert photo.state == "analyzing"


def test_bad_review_and_revision_recommendation_block_approval(workflow):
    photo = process_first(workflow)
    path = import_review(workflow, photo, recommendation="revise")
    with pytest.raises(FlowError, match="requests revision"):
        workflow.approve(photo.id, "v001", "Test user")
    bad = json.loads(path.read_text())
    bad["comparison_sha256"] = "0" * 64
    write_json(path, bad)
    with pytest.raises(FlowError, match="different image"):
        workflow.review(photo.id, path)


def test_cli_offline_handoff_and_errors(session, capsys):
    assert main(["scan", str(session.root), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["photos"]) == 1
    assert main(["next", str(session.root)]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "pending"
    assert main(["export", str(session.root), "frame"]) == 1
    assert "approved" in json.loads(capsys.readouterr().err)["error"]
    assert main(["schema", "plan"]) == 0
    assert json.loads(capsys.readouterr().out)["additionalProperties"] is False


def test_export_recovery_after_files_published_before_queue_update(workflow):
    photo = process_first(workflow)
    import_review(workflow, photo)
    workflow.approve(photo.id, "v001", "Test user")
    workflow.export(photo.id)
    q = queue.load(workflow.session)
    q.photos[0].state = "exporting"
    queue.save(workflow.session, q)
    result = Workflow(workflow.session, workflow.decoder).recover()
    assert result[0]["state"] == "exported"


def test_partial_export_preserved_for_explicit_recovery(workflow):
    photo = process_first(workflow)
    import_review(workflow, photo)
    workflow.approve(photo.id, "v001", "Test user")
    queue.transition(photo, "exporting", "simulated crash")
    partial = workflow.session.path("final", f"{photo.id}.jpg")
    partial.write_bytes(b"partial export")
    workflow.save()
    result = Workflow(workflow.session, workflow.decoder).recover()
    assert result[0]["state"] == "failed"
    assert partial.read_bytes() == b"partial export"


def test_manifest_cannot_omit_preview_checksums(workflow):
    photo = process_first(workflow)
    manifest = workflow.session.work(photo.id, "v001", "manifest.json")
    data = json.loads(manifest.read_text())
    del data["artifacts"]["after.jpg"]
    write_json(manifest, data)
    with pytest.raises(ValueError, match="missing required artifacts"):
        workflow.validate_version(photo, "v001")


def test_report_failure_does_not_publish_incomplete_version(workflow, monkeypatch):
    photo = workflow.queue.photos[0]
    workflow.prepare(photo.id)
    workflow.import_plan(photo.id, make_plan(workflow, photo))
    def fail(*args, **kwargs):
        raise OSError("simulated report disk failure")
    monkeypatch.setattr(reporting, "version_report", fail)
    with pytest.raises(OSError, match="report disk failure"):
        workflow.process(photo.id)
    persisted = queue.load(workflow.session).photos[0]
    assert persisted.state == "failed"
    assert persisted.versions == []
    assert not workflow.session.work(photo.id, "v001").exists()
    assert workflow.session.work(photo.id, ".staging-v001").is_dir()


def test_changed_feedback_after_import_blocks_processing(workflow):
    photo = process_first(workflow)
    feedback = workflow.session.work(photo.id, "v001", "feedback.md")
    feedback.write_text("First request")
    workflow.request_revision(photo.id)
    workflow.import_plan(photo.id, make_plan(workflow, photo, parent="v001", feedback=feedback))
    feedback.write_text("New request")
    with pytest.raises(FlowError, match="checksum"):
        workflow.process(photo.id)
    assert photo.current_version == "v001"


def test_approving_earlier_version_exports_exact_recipe(workflow):
    photo = process_first(workflow)
    import_review(workflow, photo)
    feedback = workflow.session.work(photo.id, "v001", "feedback.md")
    feedback.write_text("Brighter")
    workflow.request_revision(photo.id)
    workflow.import_plan(photo.id, make_plan(workflow, photo, parent="v001", exposure=0.5, feedback=feedback))
    workflow.process(photo.id)
    import_review(workflow, photo)
    workflow.approve(photo.id, "v001", "Test user choosing earlier version")
    exported = workflow.export(photo.id)
    assert exported["source_version"] == "v001"
    first_recipe = workflow.session.work(photo.id, "v001", "recipe.json")
    final_manifest = workflow.session.path("final", f"{photo.id}.manifest.json")
    assert json.loads(final_manifest.read_text())["recipe_sha256"] == digest(first_recipe)


def test_recovery_does_not_register_corrupt_published_version(workflow):
    photo = process_first(workflow)
    q = queue.load(workflow.session)
    p = q.photos[0]
    p.state, p.current_version, p.versions, p.operation_version = "processing", None, [], "v001"
    queue.save(workflow.session, q)
    workflow.session.work(photo.id, "v001", "after.jpg").write_bytes(b"corrupted output")
    recovered = Workflow(workflow.session, workflow.decoder)
    assert recovered.recover()[0]["state"] == "failed"
    p = queue.load(workflow.session).photos[0]
    assert p.versions == [] and p.current_version is None
    assert workflow.session.work(photo.id, "v001").is_dir()  # Evidence is preserved.
