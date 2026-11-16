"""Chat presentation bundles for real artifacts; no assumed Markdown image rendering."""
import json
import os
import uuid
from typing import Literal

from .models import Checksum, Identifier, StrictModel, Text, Version
from .storage import FlowError, digest, now, read_model, write_json
from . import reporting


class ChatReport(StrictModel):
    schema_version: Literal[1] = 1
    photo_id: Identifier
    filename: Text
    session: Text
    version: Version
    decision_version: Version
    decision_review_sha256: Checksum
    source_sha256: Checksum
    style: Text
    status: Literal["ready_for_user_review", "requires_user_input"]
    artifacts: dict[str, str]
    checksums: dict[str, Checksum]
    summary: list[str]
    improvements: list[str]
    limitations: list[str]
    selection_rationale: Text
    version_report: str
    bundle_path: str
    approval_required: Literal[True] = True
    frontend_rendering_verified: Literal[False] = False
    presentation_note: Text
    created_at: str


def verify_chat_report(workflow, value):
    report = ChatReport.model_validate_json(json.dumps(value))
    photo = workflow.photo(report.photo_id)
    if report.version != photo.selected_version or report.source_sha256 != photo.source_sha256:
        raise FlowError("Chat report selection/source became stale")
    folder, _, manifest = workflow.validate_version(photo, report.version)
    for relative, absolute in report.artifacts.items():
        expected = workflow.session.work(photo.id, report.version, relative)
        if (absolute != str(expected) or relative not in manifest["artifacts"]
                or digest(expected) != report.checksums.get(relative)):
            raise FlowError("Chat report references an invalid or changed image artifact")
    if not {"before.jpg", "after.jpg", "before_after.jpg"} <= report.artifacts.keys():
        raise FlowError("Chat report must reference actual before/after/comparison")
    if digest(workflow.session.work(photo.id, report.decision_version, "ai_review.json")) != report.decision_review_sha256:
        raise FlowError("Chat report visual decision became stale")
    workflow.session.source(photo)
    return report


def create_chat_report(workflow, photo):
    if not photo.selected_version:
        raise FlowError("AI must explicitly select a best version before creating its chat report")
    folder, recipe, manifest = workflow.validate_version(photo, photo.selected_version)
    decision_path = workflow.session.work(photo.id, photo.current_version, "ai_review.json")
    from .models import AIReview
    decision = read_model(decision_path, AIReview)
    if decision.best_version != photo.selected_version:
        raise FlowError("Selection does not match the latest visual AI decision")
    relatives = ["before.jpg", "after.jpg", "before_after.jpg"]
    relatives += [f"details/{r.id}/before_after.jpg" for r in recipe.detail_regions]
    directory = workflow.session.work(photo.id, "chat_reports", uuid.uuid4().hex)
    directory.mkdir(parents=True)
    report = ChatReport(photo_id=photo.id, filename=photo.filename, session=workflow.session.root.name,
        version=photo.selected_version, decision_version=photo.current_version,
        decision_review_sha256=digest(decision_path), source_sha256=photo.source_sha256,
        style=recipe.style,
        status="requires_user_input" if decision.intervention != "none" or decision.recommendation != "human_review" else "ready_for_user_review",
        artifacts={relative: str(workflow.session.work(photo.id, photo.selected_version, relative)) for relative in relatives},
        checksums={relative: manifest["artifacts"][relative] for relative in relatives},
        summary=[recipe.rationale], improvements=decision.improvements, limitations=decision.concerns,
        selection_rationale=photo.selection_rationale, version_report=str(folder / "report.md"),
        bundle_path=str(directory / "report.json"),
        presentation_note="Use available view_image/ImageContent tool on before_after.jpg and, when useful, before.jpg/after.jpg. Python does not call Codex. Client-screen rendering cannot be verified by Python; provide direct links if tool images are not visible. Local Markdown paths alone are not proof of inline chat rendering.",
        created_at=now())
    verify_chat_report(workflow, report.model_dump(mode="json"))
    def link(relative):
        return os.path.relpath(folder / relative, directory)
    content = f"""# {photo.filename} — {report.version}

Съёмка: {report.session}. Стиль: {report.style}. Статус: **{report.status}**.
Версия выбрана AI; одобрение пользователя ещё требуется. Новейшая версия: {photo.current_version}.

| ДО | ПОСЛЕ |
|---|---|
| ![До]({link('before.jpg')}) | ![После]({link('after.jpg')}) |

![Реальное сравнение]({link('before_after.jpg')})

## Обработка

{recipe.rationale}

## Оценка AI

{report.selection_rationale}

{reporting.bullets(report.improvements)}

## Ограничения

{reporting.bullets(report.limitations)}

[Постоянный отчёт версии]({link('report.md')}). Экспорт не выполнялся.

Для разговора использовать tool ImageContent, если он доступен, или прямые ссылки.
Встроенный Markdown-preview файла и inline-рендер разговора — разные возможности;
отображение на экране клиента Python не подтверждает.
"""
    reporting.text(directory / "report.md", content)
    write_json(directory / "report.json", report)
    return report.model_dump(mode="json")
