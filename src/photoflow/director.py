"""Durable Codex-assisted director. Python emits explicit actions; it never calls chat."""
import json
import uuid
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import Field

from . import imaging, queue as queues, reporting
from .cache import RawCache
from .models import AIReview, Identifier, Plan, StrictModel, Text
from .storage import FlowError, checked_id, digest, now, read_model, write_json

ANALYSIS_CATEGORIES = {"exposure", "white_balance", "highlights", "shadows", "facial_brightness",
    "skin_tones", "contrast", "background", "color_harmony", "hair_detail", "clothing_texture",
    "subject_separation", "retouching"}
REVIEW_CATEGORIES = {"facial_exposure", "skin_tones", "tonal_contrast", "background", "hair_detail",
                     "clothing_texture", "local_artifacts", "texture", "finish"}


class DirectorConfig(StrictModel):
    max_full_passes: Literal[1, 2] = 2
    max_previews: Annotated[int, Field(ge=1, le=6)] = 3
    style: Text = "Natural Commercial Portrait"


class DirectorRun(StrictModel):
    schema_version: Literal[1] = 1
    id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    instruction: Text
    config: DirectorConfig
    photo_ids: list[Identifier]
    completed: list[Identifier] = []
    active: Identifier | None = None
    baseline_versions: dict[str, list[str]] = {}
    status: Literal["active", "complete", "intervention"] = "active"
    reason: str | None = None
    pending_bundle: str | None = None
    preview_ids: dict[str, list[str]] = {}
    initialized: list[Identifier] = []
    deliveries: list[dict] = []
    events: list[dict] = []
    created_at: str
    updated_at: str


def changed_adjustments(before, after):
    changed = set()
    for group in ["raw", "global_adjustments"]:
        for key, value in getattr(before, group).model_dump().items():
            if value != getattr(after, group).model_dump()[key]:
                changed.add(f"{group}.{key}")
    for key in ["masks", "local_adjustments", "crop"]:
        if getattr(before, key) != getattr(after, key):
            changed.add(key)
    return changed


def validate_complete_plan(workflow, photo, plan):
    if photo.target_style and plan.recipe.style != photo.target_style:
        raise FlowError("Recipe style must match the requested photographic direction")
    missing = ANALYSIS_CATEGORIES - plan.analysis.coverage.keys()
    if missing:
        raise FlowError(f"Comprehensive visual analysis is incomplete: {sorted(missing)}")
    if plan.analysis.people_present and not plan.recipe.detail_regions:
        raise FlowError("Portrait plans need AI-selected native detail regions")
    if photo.current_version and photo.revision_request:
        folder, previous, _ = workflow.validate_version(photo, photo.revision_request.parent_version)
        data = json.loads((folder / "ai_review.json").read_text())
        if data.get("recommendation") == "revise" and data.get("corrections"):
            review = AIReview.model_validate_json((folder / "ai_review.json").read_bytes())
            allowed = {p for item in review.corrections for p in item.adjustment_paths}
            changes = changed_adjustments(previous, plan.recipe)
            if not changes or not changes <= allowed:
                raise FlowError(f"Correction must address only recommended adjustments: {sorted(allowed)}; changed {sorted(changes)}")


def review_context(workflow, photo):
    folder, recipe, _ = workflow.validate_version(photo, photo.current_version)
    required = {f"{photo.current_version}/{name}" for name in
                ["before.jpg", "after.jpg", "before_after.jpg", "recipe.json", "metrics.json", "session_style.json"]}
    required.update(f"{photo.current_version}/details/{r.id}/before_after.jpg" for r in recipe.detail_regions)
    return {"required_artifacts": {p: digest(workflow.session.work(photo.id, p)) for p in sorted(required)},
            "available_versions": [{"version": v, "after": str(workflow.session.work(photo.id, v, "after.jpg")),
                "after_sha256": digest(workflow.session.work(photo.id, v, "after.jpg"))} for v in photo.versions],
            "recipe": recipe.model_dump(mode="json"),
            "metrics": json.loads((folder / "metrics.json").read_text()),
            "review_categories": sorted(REVIEW_CATEGORIES), "review_schema": AIReview.model_json_schema(),
            "remaining_full_passes": photo.max_full_passes - len(photo.render_attempts),
            "instruction": "Inspect real images/details and read recipe/style/metrics. Explicitly select best_version; newest is not necessarily best. Never approve."}


def validate_directed_review(workflow, photo, review):
    context = review_context(workflow, photo)
    if REVIEW_CATEGORIES - review.assessment.keys():
        raise FlowError("Directed review must cover all visual quality categories")
    for path, checksum in context["required_artifacts"].items():
        if review.inspected_artifacts.get(path) != checksum:
            raise FlowError(f"Review must reference the actual inspected artifact: {path}")
    for relative, checksum in review.inspected_artifacts.items():
        parts = Path(relative).parts
        if len(parts) < 2 or parts[0] not in photo.versions:
            raise FlowError("Review artifact must belong to a registered processing version")
        path = workflow.session.work(photo.id, relative)
        if digest(path) != checksum:
            raise FlowError("Inspected artifact checksum mismatch")
    if review.full_resolution_texture_inspected and not context["recipe"]["detail_regions"]:
        raise FlowError("Cannot claim native texture inspection without detail crops")
    if review.best_version not in photo.versions or not review.selection_rationale.strip():
        raise FlowError("An explicit justified best-version selection is required")
    selected_path = f"{review.best_version}/after.jpg"
    if review.inspected_artifacts.get(selected_path) != digest(workflow.session.work(photo.id, selected_path)):
        raise FlowError("Selected version must have been inspected")
    if review.recommendation == "revise" and not review.corrections:
        raise FlowError("Revision requires specific structured correction recommendations")


class Director:
    def __init__(self, workflow):
        self.workflow, self.session = workflow, workflow.session
        self.path = self.session.path("director", "current.json")
        self.run = read_model(self.path, DirectorRun) if self.path.exists() else None

    def save(self):
        self.run.updated_at = now()
        validated = DirectorRun.model_validate_json(self.run.model_dump_json())
        write_json(self.path, validated)
        write_json(self.session.path("director", "history", f"{self.run.id}.json"), validated)

    def start(self, instruction, photo_id=None, config=None, next_only=False):
        if self.run and self.run.status != "complete":
            raise FlowError("A director run already exists; resume it instead of resetting its budgets")
        if photo_id:
            ids = [self.workflow.photo(photo_id).id]
        else:
            eligible = {"pending", "analyzing", "ready", "revision_requested", "processing", "failed"}
            indexed = sorted(enumerate(self.workflow.queue.photos), key=lambda pair: (-pair[1].priority, pair[0]))
            ids = [p.id for _, p in indexed if p.state in eligible and not p.duplicate_of]
            if next_only:
                ids = ids[:1]
        self.run = DirectorRun(id=uuid.uuid4().hex, instruction=instruction, config=config or DirectorConfig(),
            photo_ids=ids, baseline_versions={p.id: list(p.versions) for p in self.workflow.queue.photos if p.id in ids},
            created_at=now(), updated_at=now())
        self.save()
        return self.advance()

    def stop(self, photo, reason):
        self.run.status, self.run.reason = "intervention", reason
        self.run.events.append({"event": "intervention", "photo_id": photo.id, "reason": reason, "at": now()})
        bundle = None
        if photo.selected_version:
            from .chat_reporting import create_chat_report
            try:
                bundle = create_chat_report(self.workflow, photo)
                self.run.pending_bundle = str(Path(bundle["bundle_path"]).relative_to(self.session.root))
            except (FlowError, OSError, ValueError, KeyError):
                # Corrupt/unavailable images cannot be presented as a successful report.
                bundle = None
        self.save()
        images = [str(self.session.work(photo.id, v, "before_after.jpg")) for v in photo.versions]
        preview = self.session.work(photo.id, "analysis_preview.jpg")
        if preview.is_file():
            images.append(str(preview))
        return {"action": "requires_user_input", "photo_id": photo.id, "reason": reason,
                "best_version": photo.selected_version, "status": photo.state,
                "bundle": bundle,
                "available_images": images,
                "instruction": "Show the relevant actual images, briefly explain in Russian, ask one high-level question; no manual processing parameters."}

    def advance(self):
        try:
            return self._advance()
        except (FlowError, OSError, ValueError, KeyError) as exc:
            if self.run and self.run.active:
                return self.stop(self.workflow.photo(self.run.active), f"Unexpected processing error: {exc}")
            raise

    def _advance(self):
        if self.run is None:
            raise FlowError("Start a director run first")
        if self.run.status == "intervention":
            bundle = json.loads(self.session.path(self.run.pending_bundle).read_text()) if self.run.pending_bundle else None
            return {"action": "requires_user_input", "reason": self.run.reason, "photo_id": self.run.active, "bundle": bundle}
        remaining = [p for p in self.run.photo_ids if p not in self.run.completed]
        if not remaining:
            self.run.status = "complete"
            self.run.active = None
            self.save()
            return {"action": "complete", "completed": self.run.completed,
                    "instruction": "All results await user approval; no export performed."}
        photo = self.workflow.photo(remaining[0])
        self.run.active = photo.id
        if photo.state == "failed":
            return self.stop(photo, photo.error or "Processing failed; do not skip this photo")
        if photo.state in {"processing", "exporting"}:
            return {"action": "recover", "photo_id": photo.id, "instruction": "Recover interrupted operation before continuing; reserved passes remain charged."}
        if photo.id not in self.run.initialized:
            if len(photo.render_attempts) > self.run.config.max_full_passes:
                return self.stop(photo, "Existing pass budget already exceeds the requested lower limit")
            photo.directed = True
            photo.target_style = self.run.config.style
            # Existing versions predate this policy; attempts recorded under the policy never reset.
            photo.max_full_passes = min(photo.max_full_passes, self.run.config.max_full_passes)
            self.workflow.save()
            self.run.initialized.append(photo.id)
            replan = False
            if photo.state == "ready":
                plan = read_model(self.session.work(photo.id, "plan.json"), Plan)
                try:
                    validate_complete_plan(self.workflow, photo, plan)
                except FlowError:
                    replan = True
            if photo.versions and (photo.state == "review" or replan):
                request = self.session.path("director", "requests", f"{self.run.id}-{photo.id}.md")
                reporting.text(request, self.run.instruction)
                self.workflow.request_revision(photo.id, feedback_file=request)
            elif replan:
                self.workflow.prepare(photo.id)
        self.save()
        if len(photo.render_attempts) >= photo.max_full_passes and photo.state in {"pending", "analyzing", "ready", "revision_requested"}:
            return self.stop(photo, "Full-resolution pass budget exhausted; preserve and present best available version")
        if photo.state == "pending":
            self.workflow.prepare(photo.id)
        if photo.state in {"analyzing", "revision_requested"}:
            context = self.session.work(photo.id, "context.json")
            if not context.exists():
                if photo.state == "analyzing":
                    self.workflow.prepare(photo.id)
                else:
                    return self.stop(photo, "Original analysis inputs unavailable")
            return {"action": "targeted_correction_plan" if photo.render_attempts else "analyze_and_plan",
                    "photo_id": photo.id, "style": self.run.config.style,
                    "context": json.loads(context.read_text()), "analysis_categories": sorted(ANALYSIS_CATEGORIES),
                    "plan_schema": Plan.model_json_schema(),
                    "parent_version": self.workflow.revision_parent(photo),
                    "feedback": self.workflow.request_revision(photo.id) if photo.state == "revision_requested" else None,
                    "instruction": "Codex inspects actual RAW preview, independently creates a complete validated plan; use explore before committing to a full pass."}
        if photo.state == "ready":
            self.workflow.process(photo.id)
        if photo.state != "review":
            return self.stop(photo, f"Unexpected workflow checkpoint: {photo.state}")
        review_file = self.session.work(photo.id, photo.current_version, "ai_review.json")
        data = json.loads(review_file.read_text())
        if data.get("status") or not data.get("assessment"):
            return {"action": "visual_review", "photo_id": photo.id, **review_context(self.workflow, photo)}
        review = read_model(review_file, AIReview)
        validate_directed_review(self.workflow, photo, review)
        if photo.selected_version != review.best_version or photo.selection_rationale != review.selection_rationale:
            # Recover a crash after atomic AIReview publication but before queue metadata save.
            photo.selected_version, photo.selection_rationale = review.best_version, review.selection_rationale
            photo.history.append({"at": now(), "event": "recover_visual_selection", "version": review.best_version})
            self.workflow.save()
        if review.intervention != "none" or review.recommendation == "manual_attention":
            return self.stop(photo, review.rationale)
        if review.recommendation == "revise":
            if len(photo.render_attempts) >= photo.max_full_passes:
                return self.stop(photo, "Second pass still has significant defects: " + review.rationale)
            request = self.session.path("director", "requests", f"{self.run.id}-{photo.current_version}.md")
            reporting.text(request, "# AI visual correction plan\n\n" + review.model_dump_json(indent=2))
            self.workflow.request_revision(photo.id, feedback_file=request)
            return self.advance()
        if not self.run.pending_bundle:
            from .chat_reporting import create_chat_report
            bundle = create_chat_report(self.workflow, photo)
            self.run.pending_bundle = str(Path(bundle["bundle_path"]).relative_to(self.session.root))
            self.save()
        bundle = json.loads(self.session.path(self.run.pending_bundle).read_text())
        return {"action": "present_chat", "photo_id": photo.id, "bundle": bundle,
                "instruction": "Display real before/after with the available local image tool. Use artifact links if the client cannot show images. Then acknowledge presentation; no approval prompt between routine batch items."}

    def explore(self, identifier, plan_path, name):
        if not self.run or self.run.status != "active" or self.run.active != self.workflow.photo(identifier).id:
            raise FlowError("Preview exploration requires the active director photo")
        checked_id(name)
        photo = self.workflow.photo(identifier)
        if photo.state not in {"analyzing", "revision_requested", "ready"}:
            raise FlowError("Explore before a full render, not after a final review")
        plan = read_model(Path(plan_path), Plan)
        validate_complete_plan(self.workflow, photo, plan)
        self.session.source(photo)
        context = json.loads(self.session.work(photo.id, "context.json").read_text())
        if (plan.recipe.photo_id != photo.id or plan.recipe.source_sha256 != photo.source_sha256
                or plan.analysis.preview_sha256 != context["preview_sha256"]
                or digest(self.session.work(photo.id, "analysis_preview.jpg")) != context["preview_sha256"]
                or plan.recipe.parent_version != self.workflow.revision_parent(photo)):
            raise FlowError("Preview plan source/parent/visual input mismatch")
        attempts = self.run.preview_ids.setdefault(photo.id, [])
        if len(attempts) >= self.run.config.max_previews:
            raise FlowError("Preview exploration budget exhausted")
        folder = self.session.work(photo.id, "exploration", name)
        if folder.exists() or name in attempts:
            raise FlowError("Preview candidate already exists; refusing overwrite")
        attempts.append(name)
        self.save()  # Failed explorations also consume their bounded slot.
        folder.mkdir(parents=True)
        cache = RawCache(self.session, self.workflow.decoder)
        baseline = cache.get(photo, imaging.RawSettings())
        developed = baseline if plan.recipe.raw == imaging.RawSettings() else cache.get(photo, plan.recipe.raw)
        def reduced(pixels):
            h, w = pixels.shape[:2]
            scale = min(0.5, 1200 / max(h, w))
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            return np.stack([np.asarray(Image.fromarray(pixels[..., c]).resize(size, Image.Resampling.LANCZOS))
                             for c in range(3)], axis=-1).astype(np.float32)
        small = reduced(developed.pixels)
        neutral = reduced(baseline.pixels)
        scale = small.shape[1] / developed.pixels.shape[1]
        result = imaging.process_pixels(small, plan.recipe, spatial_scale=scale)
        imaging.save_previews(imaging.crop_image(neutral, plan.recipe.crop), result, folder)
        write_json(folder / "plan.json", plan)
        record = {"resolution": "preview_only", "full_passes_used": len(photo.render_attempts),
            "source_dimensions": developed.metadata["oriented_dimensions"], "linear_preview_dimensions": [small.shape[1], small.shape[0]],
            "spatial_radius_scale": scale, "raw_cache": cache.events, "recipe": plan.recipe.model_dump(mode="json"),
            "measurements": imaging.metrics(result), "artifacts": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}}
        write_json(folder / "manifest.json", record)
        return {"preview": str(folder / "before_after.jpg"), **record}

    def presented(self, method):
        if not self.run or self.run.status != "active" or not self.run.pending_bundle:
            raise FlowError("No ordinary chat report is awaiting presentation")
        if method not in {"tool_image_content", "artifact_links"}:
            raise FlowError("Unsupported presentation mechanism")
        bundle = json.loads(self.session.path(self.run.pending_bundle).read_text())
        photo = self.workflow.photo(self.run.active)
        if bundle["photo_id"] != photo.id:
            raise FlowError("Chat report does not belong to the active director photo")
        from .chat_reporting import verify_chat_report
        verify_chat_report(self.workflow, bundle)
        self.run.deliveries.append({"photo_id": photo.id, "selected_version": photo.selected_version,
            "bundle": self.run.pending_bundle, "method": method, "frontend_rendering_verified": False,
            "at": now(), "note": "Agent acknowledgement of artifact presentation, not user approval"})
        self.run.completed.append(photo.id)
        self.run.pending_bundle = None
        self.save()
        # Do not decode/process the next photo as a side effect of presentation acknowledgement.
        return {"presented": photo.id, "next_action": "Call direct --resume to continue within this active conversation", "approved": False}

    def resolve(self, reason):
        if not self.run or self.run.status != "intervention" or not reason.strip():
            raise FlowError("Resolution requires a blocked run and an explicit high-level user decision")
        self.run.events.append({"event": "user_resolution", "reason": reason, "at": now()})
        self.run.status, self.run.reason, self.run.pending_bundle = "active", None, None
        self.save()
        return self.advance()
