"""Transactional single-worker Codex handoff, revisions and approval-gated export."""
import json
import os
import re
import uuid
import hashlib
from time import perf_counter
from pathlib import Path

from . import __version__
from . import imaging, queue as queues, reporting
from .cache import RawCache
from .models import (AIReview, Crop, DetailRegion, ExportManifest, Plan, RawSettings, Recipe, Region,
                     ProcessingCycle, RenderAttempt, RevisionRequest, SessionStyle, VersionManifest)
from .storage import (FlowError, atomic_write, checked_version, digest, json_bytes, now,
                      read_model, sync_dir, write_json)


class Workflow:
    """Caller holds the session lock throughout every public operation."""
    def __init__(self, session, decoder=imaging.decode_raw):
        self.session = session
        self.decoder = decoder
        self.queue = queues.load(session)

    def decoded(self, photo, settings, cache=None):
        return (cache or RawCache(self.session, self.decoder)).get(photo, settings)

    def begin_cycle(self, identifier, authorized_by, reason, max_passes=2):
        """Explicit new user-directed work; never called by automatic retry/director."""
        photo = self.photo(identifier)
        director_path = self.session.path("director","current.json")
        if director_path.exists() and json.loads(director_path.read_text()).get("status") != "complete":
            raise FlowError("Finish the existing Director run before explicitly authorizing a new cycle")
        if photo.state not in {"review", "failed", "approved", "exported"} or photo.active_attempt_count < 1:
            raise FlowError("A new authorized cycle requires a completed/failed previous processing cycle")
        cycle = ProcessingCycle(start_attempt=len(photo.render_attempts),max_full_passes=max_passes,
            authorized_by=authorized_by,reason=reason,created_at=now())
        photo.processing_cycles.append(cycle)
        photo.history.append({"at":now(),"event":"user_authorized_processing_cycle",**cycle.model_dump(mode="json")})
        self.save()
        return cycle.model_dump(mode="json")

    def semantic_cache(self, photo, recipe):
        if not recipe.semantic_masks:
            return None
        from .segmentation import SemanticCache
        cache = SemanticCache(self.session,photo,recipe.segmentation.cache_key)
        for region in recipe.semantic_masks:
            if not cache.manifest.regions.get(region.region,{}).get("available"):
                raise FlowError(f"Semantic region unavailable: {region.region}; inspect diagnostics")
        return cache

    def photo(self, identifier):
        return queues.find(self.queue, identifier)

    def healing_provider(self, photo, recipe):
        from .healing import HealingProvider, authorization
        verified = authorization(self, photo, recipe)
        return HealingProvider(verified, self.semantic_cache(photo, recipe)) if verified else None

    def plan_defaults(self, plan):
        from .models import NaturalSkinRetouch
        # Revisions inherit the recorded setting, including legacy/off. Healing
        # consent cannot enable an unrelated retouch operation.
        if plan.recipe.natural_skin is None:
            if plan.recipe.parent_version:
                _, parent, _ = self.validate_version(self.photo(plan.recipe.photo_id),plan.recipe.parent_version)
                plan.recipe.natural_skin = parent.natural_skin
            else:
                plan.recipe.natural_skin = NaturalSkinRetouch()
        director=self.session.path('director','current.json')
        if director.exists():
            run=json.loads(director.read_text())
            if run.get('status')=='active' and run.get('active')==plan.recipe.photo_id and run.get('config',{}).get('smart_crop'):
                from .cropping import checkpoint,apply_selection
                selected=checkpoint(self,self.photo(plan.recipe.photo_id))
                if selected['action']!='crop_selected':
                    raise FlowError('Show crop proposals and wait for explicit user selection before importing/rendering a plan')
                plan=apply_selection(plan,selected)
        return plan

    def validate_crop(self, photo, recipe):
        from .cropping import authorization
        if recipe.approved_crop:
            return authorization(self,photo,recipe)
        previous = self.validate_version(photo,recipe.parent_version)[1] if recipe.parent_version else None
        if recipe.crop != (previous.crop if previous else Crop()):
            raise FlowError('A new crop requires explicit user approval of 16:9 or 9:16')
        if previous and previous.approved_crop:
            raise FlowError('Preserve the approved crop binding when revising this version')
        return None

    def save(self):
        queues.save(self.session, self.queue)
        reporting.session_report(self.session, self.queue)

    def fail(self, photo, exc):
        for attempt in photo.render_attempts:
            if attempt.status == "reserved":
                attempt.status, attempt.reason = "failed", str(exc)
        if photo.state != "failed":
            queues.transition(photo, "failed", str(exc))
        photo.error = str(exc)
        self.save()

    def prepare(self, identifier):
        photo = self.photo(identifier)
        if photo.state not in {"pending", "analyzing", "failed", "ready"}:
            raise FlowError(f"Cannot prepare analysis from {photo.state}; use revise for existing versions")
        if photo.versions:
            raise FlowError("Existing versions require revise, not a new initial analysis")
        if photo.state != "analyzing":
            queues.transition(photo, "analyzing", "Prepare local visual inputs; AI analysis still pending")
        self.save()
        try:
            source = self.session.source(photo)
            decoded = self.decoded(photo, RawSettings())
            folder = self.session.work(photo.id)
            folder.mkdir(parents=True, exist_ok=True)
            preview = folder / "analysis_preview.jpg"
            imaging.to_preview(decoded.pixels).save(preview, quality=93, icc_profile=imaging.srgb_profile())
            context = {"schema_version": 1, "photo_id": photo.id,
                       "source_sha256": photo.source_sha256, "preview_sha256": digest(preview),
                       "preview": str(preview), "metadata": decoded.metadata,
                       "measurements": imaging.metrics(decoded.pixels),
                       "session_style": read_model(self.session.path("session_style.json"), SessionStyle).model_dump(),
                       "ai_status": "awaiting_codex_visual_analysis",
                       "handoff": "Inspect analysis_preview.jpg, write Plan JSON, then import-plan. No Python-to-Codex API exists."}
            write_json(folder / "context.json", context)
            write_json(folder / "plan.schema.json", Plan.model_json_schema())
            return context
        except Exception as exc:
            self.fail(photo, exc)
            raise

    def import_plan(self, identifier, path):
        photo = self.photo(identifier)
        if photo.state not in {"analyzing", "revision_requested"}:
            raise FlowError("Plan import requires analyzing or revision_requested state")
        plan = self.plan_defaults(read_model(Path(path), Plan))
        if photo.directed:
            from .director import validate_complete_plan
            validate_complete_plan(self, photo, plan)
        self.session.source(photo)
        context = json.loads(self.session.work(photo.id, "context.json").read_text())
        if plan.recipe.photo_id != photo.id or plan.recipe.source_sha256 != photo.source_sha256:
            raise FlowError("Plan photo/source mismatch")
        preview = self.session.work(photo.id, "analysis_preview.jpg")
        if plan.analysis.preview_sha256 != context["preview_sha256"] or digest(preview) != context["preview_sha256"]:
            raise FlowError("Analysis is not bound to current visual input")
        if plan.recipe.parent_version != self.revision_parent(photo):
            raise FlowError("Recipe parent must match the requested revision base")
        if photo.state == "revision_requested":
            self.check_feedback(photo, plan.recipe)
        elif plan.recipe.feedback_sha256 is not None or plan.recipe.feedback_applied:
            raise FlowError("Initial recipe cannot claim feedback")
        self.semantic_cache(photo,plan.recipe)
        self.healing_provider(photo,plan.recipe)
        self.validate_crop(photo,plan.recipe)
        # Keep one coherent plan as source of truth; analysis views are derived.
        write_json(self.session.work(photo.id, "plan.json"), plan)
        write_json(self.session.work(photo.id, "analysis.json"), plan.analysis)
        reporting.text(self.session.work(photo.id, "analysis.md"), reporting.analysis_markdown(plan.analysis))
        queues.transition(photo, "ready", f"Validated visual plan imported: {plan.analysis.provider}/{plan.analysis.model}")
        self.save()
        return {"photo_id": photo.id, "state": photo.state, "provider": plan.analysis.provider}

    def check_feedback(self, photo, recipe):
        feedback = self.feedback_path(photo)
        if not recipe.feedback_applied.strip() or recipe.feedback_sha256 != digest(feedback):
            raise FlowError("Revision must explain feedback and reference the current feedback.md checksum")

    def revision_parent(self, photo):
        return photo.revision_request.parent_version if photo.revision_request else photo.current_version

    def feedback_path(self, photo):
        relative = (photo.revision_request.feedback_path if photo.revision_request
                    else f"{photo.current_version}/feedback.md")
        return self.session.work(photo.id, relative)

    def version_folder(self, photo, version):
        checked_version(version)
        if version not in photo.versions:
            raise FlowError("Version is not registered")
        return self.session.work(photo.id, version)

    def validate_version(self, photo, version):
        folder = self.version_folder(photo, version)
        manifest = read_model(folder / "manifest.json", VersionManifest).model_dump(mode="json")
        if manifest["status"] != "complete" or manifest["version"] != version or manifest["source_sha256"] != photo.source_sha256:
            raise FlowError("Version manifest identity invalid")
        for relative, checksum in manifest["artifacts"].items():
            path = self.session.work(photo.id, version, relative)
            if digest(path) != checksum:
                raise FlowError(f"Version artifact modified: {relative}")
        recipe = read_model(folder / "recipe.json", Recipe)
        if recipe.photo_id != photo.id or recipe.source_sha256 != photo.source_sha256 or digest(folder / "recipe.json") != manifest["recipe_sha256"]:
            raise FlowError("Version recipe identity invalid")
        if recipe.parent_version != manifest["parent_version"] or recipe.masks != [Region.model_validate(m) for m in manifest["masks"]]:
            raise FlowError("Version recipe/mask manifest mismatch")
        if recipe.detail_regions != [DetailRegion.model_validate(r) for r in manifest["detail_regions"]]:
            raise FlowError("Version recipe/detail manifest mismatch")
        if [m.model_dump(mode="json") for m in recipe.semantic_masks] != manifest["semantic_masks"]:
            raise FlowError("Version semantic mask manifest mismatch")
        if (recipe.segmentation.model_dump(mode="json") if recipe.segmentation else None) != manifest["segmentation"]:
            raise FlowError("Version segmentation manifest mismatch")
        if (recipe.spot_healing.model_dump(mode="json") if recipe.spot_healing else None) != manifest["spot_healing"]:
            raise FlowError("Version healing consent manifest mismatch")
        if (recipe.natural_skin.model_dump(mode="json") if recipe.natural_skin else None) != manifest["natural_skin"]:
            raise FlowError("Version skin retouch manifest mismatch")
        if (recipe.approved_crop.model_dump(mode="json") if recipe.approved_crop else None) != manifest['approved_crop']:
            raise FlowError('Version crop approval manifest mismatch')
        for name in ["report.md", "feedback.md", "ai_review.json"]:
            if not self.session.work(photo.id, version, name).is_file():
                raise FlowError(f"Mandatory version artifact missing: {name}")
        return folder, recipe, manifest

    def next_version(self, photo):
        root = self.session.work(photo.id)
        numbers = [int(v[1:]) for v in photo.versions]
        for folder in root.iterdir():
            match = re.fullmatch(r"(?:\.staging-)?v([0-9]{3,})", folder.name)
            if match:
                numbers.append(int(match.group(1)))
        return f"v{max(numbers, default=0) + 1:03d}"

    def process(self, identifier):
        photo = self.photo(identifier)
        if photo.state != "ready":
            raise FlowError("Processing requires a validated imported plan (state ready)")
        if photo.active_attempt_count >= photo.active_pass_limit:
            raise FlowError("Full-resolution pass limit reached; stop and present the best available version")
        original_photo = photo.model_copy(deep=True)
        plan = self.plan_defaults(read_model(self.session.work(photo.id, "plan.json"), Plan))
        if photo.directed:
            from .director import validate_complete_plan
            validate_complete_plan(self, photo, plan)
        recipe = plan.recipe
        if recipe.parent_version != self.revision_parent(photo):
            raise FlowError("Stale plan parent version")
        if recipe.photo_id != photo.id or recipe.source_sha256 != photo.source_sha256:
            raise FlowError("Plan identity changed")
        if recipe.parent_version:
            self.validate_version(photo, recipe.parent_version)
            self.check_feedback(photo, recipe)
        semantics = self.semantic_cache(photo,recipe)
        healing = self.healing_provider(photo,recipe)
        crop_consent = self.validate_crop(photo,recipe)
        version = self.next_version(photo)
        temporary = self.session.work(photo.id, f".staging-{version}")
        destination = self.session.work(photo.id, version)
        if destination.exists():
            raise FlowError("Version collision")
        temporary.mkdir()
        photo.operation_version = version
        photo.render_attempts.append(RenderAttempt(version=version,
            recipe_sha256=hashlib.sha256(json_bytes(recipe)).hexdigest(), started_at=now()))
        queues.transition(photo, "processing", f"Reserved {version}")
        self.save()
        try:
            started = perf_counter()
            cache = RawCache(self.session, self.decoder)
            source = self.session.source(photo)
            baseline = self.decoded(photo, RawSettings(), cache)
            # Reuse baseline only if RAW settings are exactly the neutral baseline.
            if recipe.raw == RawSettings():
                decoded = baseline
            else:
                decoded = self.decoded(photo, recipe.raw, cache)
            write_json(temporary / "recipe.json", recipe)
            write_json(temporary / "analysis.json", plan.analysis)
            write_json(temporary / "decoder.json", decoded.metadata)
            style = read_model(self.session.path("session_style.json"), SessionStyle)
            write_json(temporary / "session_style.json", style)
            if crop_consent:
                for name in ['proposal.json','decision.json']:
                    atomic_write(temporary/'cropping'/name,(crop_consent/name).read_bytes())
            if semantics:
                if decoded.metadata["oriented_dimensions"] != semantics.manifest.dimensions:
                    raise FlowError("Segmentation/RAW oriented dimensions mismatch")
                write_json(temporary / "segmentation.json",semantics.manifest)
            result = imaging.process_pixels(decoded.pixels, recipe, temporary / "masks",semantic_provider=semantics,
                healing_provider=healing, healing_dir=temporary / "healing" if healing else None)
            before = imaging.crop_image(baseline.pixels, recipe.crop,recipe.approved_crop)
            measures = {"before": imaging.metrics(before), "after": imaging.metrics(result)}
            measures.update(imaging.save_previews(before, result, temporary))
            measures["details"] = imaging.save_details(before, result, baseline.pixels.shape, recipe, temporary)
            requested_skin = recipe.natural_skin is not None and recipe.natural_skin.enabled
            available_skin = semantics is not None and semantics.manifest.regions.get('face_skin',{}).get('available',False)
            measures['natural_skin'] = {'enabled':requested_skin,
                'status':'applied' if requested_skin and available_skin else 'off' if not requested_skin else 'unavailable',
                'note':'Chroma field only; luminance texture retained. No missing-mask fallback.'}
            measures["performance"] = {"elapsed_before_publish_seconds": perf_counter() - started,
                                       "raw_cache": cache.events, "full_pass": photo.active_attempt_count,
                                       "max_full_passes": photo.active_pass_limit,
                                       "lifetime_attempts": len(photo.render_attempts)}
            write_json(temporary / "metrics.json", measures)
            reporting.text(temporary / "feedback.md", f"# Обратная связь — {photo.filename} {version}\n\nStatus: review\n\n## Запрошенные изменения\n\n")
            write_json(temporary / "ai_review.json", {"status": "awaiting_codex_visual_review"})
            if recipe.parent_version:
                # Exact historical feedback snapshot, independent of future user edits.
                reporting.text(temporary / "feedback_applied.md",
                               self.feedback_path(photo).read_text())
            self.session.source(photo)  # Verify RAW integrity again after all processing.
            mutable = {"feedback.md", "ai_review.json", "report.md", "manifest.json"}
            artifacts = {str(p.relative_to(temporary)): digest(p) for p in sorted(temporary.rglob("*"))
                         if p.is_file() and p.name not in mutable}
            manifest = {"schema_version": 1, "status": "complete", "version": version,
                        "source_filename": photo.filename, "source_sha256": photo.source_sha256,
                        "parent_version": recipe.parent_version, "recipe_sha256": digest(temporary / "recipe.json"),
                        "engine_version": __version__, "libraries": imaging.runtime_versions(),
                        "ai": {"provider": plan.analysis.provider, "model": plan.analysis.model},
                        "masks": [m.model_dump() for m in recipe.masks], "created_at": now(),
                        "semantic_masks": [m.model_dump() for m in recipe.semantic_masks],
                        "segmentation": recipe.segmentation.model_dump() if recipe.segmentation else None,
                        "spot_healing": recipe.spot_healing.model_dump() if recipe.spot_healing else None,
                        "natural_skin": recipe.natural_skin.model_dump() if recipe.natural_skin else None,
                        'approved_crop':recipe.approved_crop.model_dump() if recipe.approved_crop else None,
                        "detail_regions": [r.model_dump() for r in recipe.detail_regions],
                        "artifacts": artifacts, "mutable_artifacts": sorted(mutable - {"manifest.json"})}
            write_json(temporary / "manifest.json", VersionManifest.model_validate_json(json.dumps(manifest)))
            # Publish only after all mandatory artifacts, including report, exist.
            photo.versions.append(version)
            photo.current_version = version
            photo.approved_version = photo.approved_recipe_sha256 = photo.approved_by = photo.approved_at = None
            queues.transition(photo, "review", f"{version} processed; visual AI review pending")
            reporting.version_report(self.session, photo, version, folder=temporary)
            os.rename(temporary, destination)
            sync_dir(destination.parent)
            photo.operation_version = None
            photo.revision_request = None
            photo.render_attempts[-1].status = "complete"
            self.save()
            return {"photo_id": photo.id, "version": version, "state": photo.state,
                    "report": str(destination / "report.md"), "ai_review": "pending"}
        except Exception as exc:
            if not destination.exists():
                # Roll back unpublished version pointers, retaining the attempt's audit history.
                for field in ["versions", "current_version", "approved_version",
                              "approved_recipe_sha256", "approved_by", "approved_at"]:
                    setattr(photo, field, getattr(original_photo, field))
            self.fail(photo, exc)
            raise

    def request_revision(self, identifier, from_version=None, feedback_file=None):
        photo = self.photo(identifier)
        if not photo.current_version:
            raise FlowError("No version to revise")
        base = from_version or self.revision_parent(photo)
        self.validate_version(photo, base)
        if photo.state not in {"review", "approved", "exported", "failed", "revision_requested", "ready"}:
            raise FlowError("Cannot request revision from current state")
        if from_version or feedback_file or photo.revision_request is None:
            feedback_relative = f"{base}/feedback.md"
            if feedback_file is not None:
                content = Path(feedback_file).read_bytes()
                if not content.decode("utf-8").strip():
                    raise FlowError("Feedback file must contain a request")
                feedback_relative = f"revision_requests/{uuid.uuid4().hex}.md"
                atomic_write(self.session.work(photo.id, feedback_relative), content)
            photo.revision_request = RevisionRequest(parent_version=base,
                                                     feedback_path=feedback_relative, requested_at=now())
        if photo.state != "revision_requested":
            queues.transition(photo, "revision_requested", "User/Codex requested a revision; interpret persisted feedback")
        self.save()
        feedback = self.feedback_path(photo)
        return {"photo_id": photo.id, "parent_version": base,
                "feedback_file": str(feedback), "feedback_sha256": digest(feedback),
                "feedback": feedback.read_text(),
                "previous_recipe": str(self.session.work(photo.id, base, "recipe.json")),
                "instruction": "Inspect current result, interpret feedback, write revised Plan and import-plan; no automatic NLP in Python"}

    def review(self, identifier, path=None):
        photo = self.photo(identifier)
        if not photo.current_version:
            raise FlowError("No processed version to review")
        folder, _, _ = self.validate_version(photo, photo.current_version)
        comparison = folder / "before_after.jpg"
        if path is not None:
            if photo.state != "review":
                raise FlowError("Import visual review only while awaiting review")
            assessment = read_model(Path(path), AIReview)
            if (assessment.photo_id, assessment.version, assessment.comparison_sha256) != (photo.id, photo.current_version, digest(comparison)):
                raise FlowError("Review references a different image or version")
            if photo.directed:
                from .director import validate_directed_review
                validate_directed_review(self, photo, assessment)
            if assessment.best_version is not None:
                self.validate_version(photo, assessment.best_version)
                if not assessment.selection_rationale.strip():
                    raise FlowError("Best-version selection requires a rationale")
                photo.selected_version = assessment.best_version
                photo.selection_rationale = assessment.selection_rationale
            write_json(folder / "ai_review.json", assessment)
            photo.history.append({"at": now(), "event": "visual_review", "provider": assessment.provider,
                                  "model": assessment.model, "recommendation": assessment.recommendation})
            self.save()
        reporting.version_report(self.session, photo, photo.current_version)
        return {"photo_id": photo.id, "version": photo.current_version,
                "comparison": str(comparison), "comparison_sha256": digest(comparison),
                "report": str(folder / "report.md"), "review_schema": AIReview.model_json_schema()}

    def approve(self, identifier, version, actor):
        photo = self.photo(identifier)
        if photo.state not in {"review", "failed"}:
            raise FlowError("Approval requires a reviewed version")
        if not actor.strip():
            raise FlowError("Explicit approving user identity required (--by)")
        folder, _, manifest = self.validate_version(photo, version)
        try:
            review = read_model(folder / "ai_review.json", AIReview)
        except ValueError as exc:
            raise FlowError("Visual AI review missing; inspect comparison and import-review first") from exc
        if (review.photo_id, review.version, review.comparison_sha256) != (photo.id, version, digest(folder / "before_after.jpg")):
            raise FlowError("Review provenance mismatch")
        if review.recommendation == "revise":
            raise FlowError("AI review requests revision; resolve before approval")
        photo.approved_version = version
        photo.approved_recipe_sha256 = manifest["recipe_sha256"]
        photo.approved_at, photo.approved_by = now(), actor
        queues.transition(photo, "approved", f"Explicit user approval of {version} by {actor}")
        self.save()
        reporting.version_report(self.session, photo, version)
        return {"photo_id": photo.id, "approved_version": version, "approved_by": actor}

    def export(self, identifier):
        photo = self.photo(identifier)
        if photo.state != "approved":
            raise FlowError("Only explicitly approved photos can be exported")
        version = photo.approved_version
        folder, recipe, manifest = self.validate_version(photo, version)
        if digest(folder / "recipe.json") != photo.approved_recipe_sha256:
            raise FlowError("Approved recipe checksum mismatch")
        if manifest["libraries"] != imaging.runtime_versions():
            raise FlowError("Library versions changed since approval; create and review a new version")
        healing = self.healing_provider(photo, recipe)
        self.validate_crop(photo,recipe)
        names = [f"{photo.id}.jpg", f"{photo.id}.tif", f"{photo.id}.manifest.json"]
        destinations = [self.session.path("final", name) for name in names]
        if any(p.exists() for p in destinations):
            raise FlowError("Final export already exists; refusing overwrite (archive explicitly before retry)")
        staging = self.session.work(photo.id, f".export-{version}")
        if staging.exists():
            raise FlowError("Interrupted export staging exists; inspect/archive it before retry")
        staging.mkdir()
        queues.transition(photo, "exporting", f"Export approved {version} from original RAW")
        self.save()
        linked = []
        try:
            source = self.session.source(photo)
            decoded = self.decoded(photo, recipe.raw)
            result = imaging.process_pixels(decoded.pixels, recipe,semantic_provider=self.semantic_cache(photo,recipe),
                                            healing_provider=healing)
            validation = imaging.export_pixels(result, staging / names[0], staging / names[1], recipe.output.jpeg_quality)
            self.session.source(photo)
            export_manifest = {"schema_version": 1, "photo_id": photo.id, "source_sha256": photo.source_sha256,
                               "source_version": version, "recipe_sha256": photo.approved_recipe_sha256,
                               "approved_by": photo.approved_by, "approved_at": photo.approved_at,
                               "created_at": now(), "validation": validation,
                               "libraries": imaging.runtime_versions(),
                               "files": {name: digest(staging / name) for name in names[:2]}}
            write_json(staging / names[2], ExportManifest.model_validate_json(json.dumps(export_manifest)))
            # Atomic no-clobber publication per file; manifest is the transaction marker.
            for name, destination in zip(names, destinations):
                os.link(staging / name, destination)
                linked.append(destination)
            sync_dir(self.session.path("final"))
            queues.transition(photo, "exported", f"Validated JPEG and 16-bit TIFF from {version}")
            self.save()
            reporting.version_report(self.session, photo, version)
            return {"photo_id": photo.id, "source_version": version,
                    "files": [str(p) for p in destinations], "validation": validation}
        except Exception as exc:
            # Only remove files that this operation itself created. Never remove pre-existing finals.
            for path in linked:
                path.unlink()
            self.fail(photo, exc)
            raise

    def recover(self):
        recovered = []
        for photo in self.queue.photos:
            if photo.state not in {"processing", "exporting"}:
                continue
            if photo.state == "processing":
                version = photo.operation_version
                complete = self.session.work(photo.id, version) if version else None
                if complete and complete.is_dir() and (complete / "manifest.json").exists():
                    candidate = photo.model_copy(deep=True)
                    if version not in candidate.versions:
                        candidate.versions.append(version)
                    try:
                        self.session.source(photo)
                        self.validate_version(candidate, version)
                    except (FlowError, OSError, ValueError, KeyError):
                        self.fail(photo, FlowError("Interrupted version is invalid; inspect artifacts before retry"))
                    else:
                        photo.versions = candidate.versions
                        photo.current_version = version
                        photo.revision_request = None
                        for attempt in photo.render_attempts:
                            if attempt.version == version:
                                attempt.status = "complete"
                        photo.approved_version = photo.approved_recipe_sha256 = photo.approved_by = photo.approved_at = None
                        queues.transition(photo, "review", "Recovered complete immutable version after interrupted queue update")
                        reporting.version_report(self.session, photo, version)
                else:
                    self.fail(photo, FlowError("Interrupted processing; staging preserved; retry via analyze or revise"))
            else:
                export = self.session.path("final", f"{photo.id}.manifest.json")
                try:
                    self.session.source(photo)
                    folder, _, _ = self.validate_version(photo, photo.approved_version)
                    if digest(folder / "recipe.json") != photo.approved_recipe_sha256:
                        raise FlowError("Approved recipe changed during interrupted export")
                    record = read_model(export, ExportManifest).model_dump(mode="json")
                    if (record["photo_id"] != photo.id or record["source_sha256"] != photo.source_sha256
                            or record["source_version"] != photo.approved_version or record["recipe_sha256"] != photo.approved_recipe_sha256):
                        raise FlowError("Export manifest mismatches approval")
                    for name, checksum in record["files"].items():
                        if digest(self.session.path("final", name)) != checksum:
                            raise FlowError("Export checksum mismatch")
                    imaging.validate_export(self.session.path("final", f"{photo.id}.jpg"),
                                            self.session.path("final", f"{photo.id}.tif"), record["validation"]["dimensions"])
                    queues.transition(photo, "exported", "Recovered validated export after interrupted queue update")
                except (OSError, ValueError, KeyError, FlowError) as exc:
                    self.fail(photo, FlowError(f"Interrupted export; inspect final/ and .export staging before retry: {exc}"))
            photo.operation_version = None
            recovered.append({"photo_id": photo.id, "state": photo.state, "error": photo.error})
        self.save()
        return recovered
