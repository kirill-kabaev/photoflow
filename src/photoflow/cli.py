"""Local CLI and machine-readable Codex tools. No web service or uploads."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from . import queue as queues, reporting
from .models import AIReview, Plan, Recipe
from .storage import FlowError, Session
from .workflow import Workflow


def parser():
    root = argparse.ArgumentParser(prog="photoflow", description="Local Codex-assisted RAW workflow")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ["scan", "status", "queue", "next", "report", "recover", "run"]:
        command = sub.add_parser(name)
        command.add_argument("session", help="Path to an existing session containing raw/")
        command.add_argument("--json", action="store_true", help="Structured JSON (also the default)")
        if name == "run":
            command.add_argument("--next", action="store_true", required=True)
    for name in ["analyze", "process", "review", "revise", "approve", "export", "import-plan", "import-review", "priority"]:
        command = sub.add_parser(name)
        command.add_argument("session")
        command.add_argument("photo", help="Stable ID or unambiguous ASCII source stem")
        command.add_argument("--json", action="store_true")
        if name in {"import-plan", "import-review"}:
            command.add_argument("--file", type=Path, required=True)
        if name == "approve":
            command.add_argument("--version", required=True)
            command.add_argument("--by", required=True, help="Approving user; only after their explicit request")
        if name == "priority":
            command.add_argument("--value", type=int, required=True)
            command.add_argument("--reason", required=True)
    schema = sub.add_parser("schema")
    schema.add_argument("kind", choices=["plan", "recipe", "review"])
    return root


def execute(args):
    if args.command == "schema":
        return {"plan": Plan, "recipe": Recipe, "review": AIReview}[args.kind].model_json_schema()
    session = Session(args.session)
    with session.lock():
        if args.command == "scan":
            queue = queues.scan(session)
            reporting.session_report(session, queue)
            return queue.model_dump(mode="json")
        workflow = Workflow(session)
        if args.command in {"status", "queue"}:
            if args.command == "queue":
                return workflow.queue.model_dump(mode="json")
            return {"total": len(workflow.queue.photos),
                    "states": dict(Counter(p.state for p in workflow.queue.photos)),
                    "duplicates": [{"id": p.id, "duplicate_of": p.duplicate_of} for p in workflow.queue.photos if p.duplicate_of],
                    "failed": [{"id": p.id, "error": p.error} for p in workflow.queue.photos if p.state == "failed"]}
        if args.command in {"next", "run"}:
            photo = queues.next_photo(workflow.queue)
            if photo is None:
                return {"next": None, "message": "No actionable photo; inspect status for review/failed checkpoints"}
            if args.command == "next":
                return photo.model_dump(mode="json")
            if photo.state in {"pending", "analyzing"}:
                return workflow.prepare(photo.id)
            if photo.state == "ready":
                return workflow.process(photo.id)
            return workflow.request_revision(photo.id)
        if args.command == "report":
            for photo in workflow.queue.photos:
                for version in photo.versions:
                    workflow.validate_version(photo, version)
                    reporting.version_report(session, photo, version)
            reporting.session_report(session, workflow.queue)
            return {"report": str(session.path("review.md"))}
        if args.command == "recover":
            return workflow.recover()
        if args.command == "analyze":
            return workflow.prepare(args.photo)
        if args.command == "process":
            return workflow.process(args.photo)
        if args.command == "import-plan":
            return workflow.import_plan(args.photo, args.file)
        if args.command == "import-review":
            return workflow.review(args.photo, args.file)
        if args.command == "review":
            return workflow.review(args.photo)
        if args.command == "revise":
            return workflow.request_revision(args.photo)
        if args.command == "approve":
            return workflow.approve(args.photo, args.version, args.by)
        if args.command == "export":
            return workflow.export(args.photo)
        if args.command == "priority":
            photo = workflow.photo(args.photo)
            queues.set_priority(photo, args.value, args.reason)
            workflow.save()
            return photo.model_dump(mode="json")
        raise FlowError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except (FlowError, ValidationError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
