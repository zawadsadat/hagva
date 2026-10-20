#!/usr/bin/env python3
"""Rebase historical scripts into a new working directory; execute no experiments.

The committed phase snapshots and supplied data/framework directories are read
only during preparation. Recorded result tables are never copied as new output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


PHASES = ("phase1", "phase2", "phase3a", "phase3b")
REPO = Path(__file__).resolve().parents[1]


def safe_path(value):
    path = Path(value).expanduser().resolve()
    # Historical shell drivers have unquoted expansions. Reject paths they cannot
    # represent, rather than inserting shell syntax into generated code.
    if not re.fullmatch(r"[/A-Za-z0-9_.+\-]+", str(path)):
        raise argparse.ArgumentTypeError("use an absolute path without spaces or shell metacharacters")
    return path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=safe_path, required=True, help="new directory, outside repository and inputs")
    parser.add_argument("--data-root", type=safe_path, required=True, help="low pools, sources and VPR-Query-NEW/ live here")
    parser.add_argument("--render-root", type=safe_path, required=True, help="wide pools and regenerated warp/mask folders")
    parser.add_argument("--framework", type=safe_path, required=True, help="dedicated scratch VPR framework checkout")
    parser.add_argument("--vpr-python", type=safe_path, required=True)
    parser.add_argument("--iqa-python", type=safe_path, required=True)
    args = parser.parse_args()
    protected = (REPO, args.data_root, args.render_root, args.framework,
                 args.vpr_python, args.iqa_python)
    if args.workspace.exists():
        parser.error("workspace already exists; choose a new directory to preserve previous outputs")
    for path in protected:
        if not path.exists():
            parser.error(f"input does not exist: {path}")
        if args.workspace == path or args.workspace in path.parents or path in args.workspace.parents:
            parser.error(f"workspace must not overlap repository or inputs: {path}")
    for path in (args.data_root, args.render_root, args.framework):
        if not path.is_dir():
            parser.error(f"expected a directory: {path}")
    for path in (args.vpr_python, args.iqa_python):
        if not path.is_file():
            parser.error(f"expected a Python executable: {path}")

    # Verify the snapshot before making a working copy.
    snapshot = json.loads((REPO / "docs/phase_snapshot_manifest.json").read_text())
    for record in snapshot["files"]:
        source = REPO / record["path"]
        if sha256(source.read_bytes()) != record["sha256"]:
            parser.error(f"snapshot differs from recorded source: {source}")

    support = args.workspace / "repository"
    replacements = {
        "/mnt/sda/HAGVA_R2/phase1/iqa_venv/bin/python3": str(args.iqa_python),
        "/home/newview/miniconda3/envs/vprtutorial/bin/python3": str(args.vpr_python),
        "/mnt/sda/Zawad/genwarp/VPR-GIT": str(args.framework),
        "/mnt/sda/Zawad/genwarp": str(args.data_root),
        "/home/newview/genwarp": str(args.render_root),
        "/mnt/sda/HAGVA_R2": str(args.workspace),
        "/mnt/sda/hagva-repo": str(support),
        "/tmp/": str(args.workspace / "scratch") + "/",
    }
    # One substitution pass prevents a replacement path being rewritten again.
    pattern = re.compile("|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
    copies = []
    for record in snapshot["files"]:
        rel = Path(record["path"])
        # Inputs and scripts only; do not seed a fresh run with recorded AUC rows,
        # cached gate outputs, or manifests pointing to absent generated images.
        if (rel.suffix in (".py", ".sh") or rel == Path("phase1/checkpoint2_verification.csv") or
                (len(rel.parts) > 2 and rel.parts[1] in ("scores", "draws", "strata"))):
            copies.append((REPO / rel, args.workspace / rel))
    for source in sorted((REPO / "module_a/scores").glob("*_400_*.csv")):
        copies.append((source, support / source.relative_to(REPO)))
    copies.append((REPO / "tools/gt_audit.py", support / "tools/gt_audit.py"))

    args.workspace.mkdir(parents=True)
    (args.workspace / "scratch").mkdir()
    for phase in PHASES:
        (args.workspace / phase / "logs").mkdir(parents=True)
    manifest = {"replacements": replacements, "files": []}
    for source, target in copies:
        original = source.read_bytes()
        data = pattern.sub(lambda match: replacements[match.group()], original.decode()).encode() if source.suffix in (".py", ".sh") else original
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        shutil.copymode(source, target)
        manifest["files"].append({"source": str(source.relative_to(REPO)),
                                  "output": str(target.relative_to(args.workspace)),
                                  "source_sha256": sha256(original), "output_sha256": sha256(data)})
    (args.workspace / "workspace_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(copies)} files in {args.workspace}; no experiments executed.")
    print("Apply patches/ to a dedicated framework copy; provision input images, GT and causal variants before running.")
    print("See docs/PHASE_REPRODUCTION.md for full gate coverage and historical limitations.")


if __name__ == "__main__":
    main()
