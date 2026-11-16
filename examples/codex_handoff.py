"""Print an intentionally invalid template; Codex must supply real visual judgments.

Usage: python examples/codex_handoff.py SESSION/work/PHOTO/context.json > /tmp/plan.json
The template cannot be imported until visually_inspected and analysis are completed.
No AI is called and no fabricated analysis is inserted.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
    args = parser.parse_args()
    context = json.loads(args.context.read_text())
    template = {
        "analysis": {
            "photo_id": context["photo_id"], "source_sha256": context["source_sha256"],
            "preview_sha256": context["preview_sha256"], "provider": "codex", "model": "",
            "visually_inspected": False, "category": "other", "people_present": False,
            "observations": [], "technical": {}, "portrait": [],
            "recommendations": [], "uncertainties": [], "human_attention": [],
        },
        "recipe": {
            "photo_id": context["photo_id"], "source_sha256": context["source_sha256"],
            "parent_version": None, "style": "", "rationale": "",
            "global_adjustments": {}, "masks": [], "local_adjustments": [],
        },
    }
    print(json.dumps(template, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
