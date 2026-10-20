#!/usr/bin/env python3
"""Check imported hashes, script syntax and score/result consistency offline."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
DATASETS = {"GardensPoint": "gardenspoint", "SFU": "sfu", "StLucia": "stlucia"}
RESULTS = {
    "phase1/phase1_results.csv": 309,
    "phase2/phase2_results.csv": 196,
    "phase3a/phase3a_gate_results.csv": 93,
    "phase3a/phase3a_causal_results.csv": 98,
    "phase3b/phase3b_gate_results.csv": 186,
    "phase3b/phase3b_causal_results.csv": 126,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        result = list(reader)
        require(all(None not in row and all(v is not None for v in row.values()) for row in result),
                f"malformed CSV: {path}")
        return result


def main():
    manifest = json.loads((REPO / "docs/phase_snapshot_manifest.json").read_text())
    scripts = 0
    for record in manifest["files"]:
        path = REPO / record["path"]
        data = path.read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                f"snapshot changed: {path}")
        if path.suffix == ".py":
            ast.parse(data, filename=str(path)); scripts += 1
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True); scripts += 1

    scores = {}
    provenance = json.loads((REPO / "phase1/scores/provenance.json").read_text())
    for dataset, record in provenance['inputs'].items():
        path = REPO / 'module_a/scores' / Path(record['combined_csv']).name
        data = path.read_bytes()
        # Original score files used CRLF; Git stored the legacy CSVs with LF.
        # Check both representations without editing either file's contents.
        crlf = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        require(record['sha256'] in {hashlib.sha256(data).hexdigest(), hashlib.sha256(crlf).hexdigest()},
                f"legacy input score hash mismatch (including newline normalization): {path}")
    for regime in ("low", "wide"):
        for dataset, key in DATASETS.items():
            relative = f"phase1/scores/{key}_400_extended.csv" if regime == "low" else f"phase3b/scores/{key}_wide_extended.csv"
            pool = rows(REPO / relative)
            require(len(pool) == len({r['image_stem'] for r in pool}) == 400, f"invalid pool: {relative}")
            for row in pool:
                for col in ("w100", "w000f", "w050f", "clipiqa_neg", "brisque"):
                    require(math.isfinite(float(row[col])), f"nonfinite {relative}: {col}")
                require(float(row["w100"]) == float(row["mean_sem_score"]), f"semantic mismatch: {relative}")
                require(float(row["w000f"]) == float(row["geo_fixed"]), f"geometry mismatch: {relative}")
                require(abs(float(row["w050f"]) - (float(row["w100"]) + float(row["w000f"])) / 2) < 1e-12,
                        f"fusion mismatch: {relative}")
            for seed in range(5):
                require(sorted(int(r[f"rand_s{seed}"]) for r in pool) == list(range(400)), f"invalid random ranks: {relative}")
            scores[regime, dataset] = pool
            if regime == 'low':
                legacy = {r['image_stem']: r for r in rows(REPO / 'module_a/scores' / f'{key}_400_combined_summary.csv')}
                require(set(legacy) == {r['image_stem'] for r in pool}, f"legacy IDs differ: {relative}")
                for row in pool:
                    require(all(row[col] == value for col, value in legacy[row['image_stem']].items()),
                            f"legacy score values differ: {relative}: {row['image_stem']}")

    checked = 0
    short = []
    main_methods = {"w100", "w000f", "w050f", "clipiqa_neg", "brisque"} | {f"rand_s{s}" for s in range(5)}
    for relative, expected in RESULTS.items():
        table = rows(REPO / relative)
        require(len(table) == expected, f"row count: {relative}: {len(table)} != {expected}")
        for row in table:
            require(math.isfinite(float(row['auc'])), f"invalid AUC: {relative}")
        if "gate_results" not in relative and relative != "phase1/phase1_results.csv":
            continue
        regime = "wide" if relative.startswith("phase3b/") else "low"
        seen = set()
        for row in table:
            identity = (row.get("descriptor", "PatchNetVLAD"), row['dataset'], row['method'], row['n_requested'])
            require(identity not in seen, f"duplicate gate cell: {relative}: {identity}")
            seen.add(identity)
            pool = scores[regime, row['dataset']]
            passed = sum(float(score[row['method']]) < float(row['tau']) for score in pool)
            require(passed == int(row['n_passed']), f"cutoff/count mismatch: {relative}: {identity}: {passed}")
            key = {'GardensPoint': 'gp', 'SFU': 'sfu', 'StLucia': 'stl'}[row['dataset']]
            prefix = '' if relative.startswith('phase1/') else identity[0] + '_'
            if regime == 'wide':
                prefix = 'wide_' + prefix
            raw = rows(REPO / relative.split('/')[0] / 'raw' / f"{prefix}{key}_{row['method']}_n{row['n_requested']}.csv")
            require(len(raw) == 1 and int(raw[0]['n_passed']) == passed and float(raw[0]['auc']) == float(row['auc']),
                    f"raw result/AUC mismatch: {relative}: {identity}")
            checked += 1
            if row['method'] in main_methods and passed != int(row['n_requested']):
                short.append([regime, *identity, str(passed)])
    require(len(short) == 12, f"expected 12 recorded count-short main-comparison rows; found {len(short)}")

    for phase, regime in (("phase2", "low"), ("phase3b", "wide")):
        for key, dataset in (("gp", "GardensPoint"), ("sfu", "SFU"), ("stl", "StLucia")):
            valid = {r['image_stem'] for r in scores[regime, dataset]}
            for seed in range(5):
                path = REPO / phase / "draws" / f"{key}_seed{seed}.csv"
                draw = rows(path)
                require(len(draw) == len({r['stem'] for r in draw}) == 50, f"invalid draw: {path}")
                require({r['stem'] for r in draw} <= valid, f"draw outside pool: {path}")

    print(f"Verified {len(manifest['files'])} snapshot hashes, {scripts} script syntax checks, six 400-view score pools,")
    print(f"six result tables, {checked} gate cutoff/count checks, and thirty 50-view draws.")
    print("Recorded main-comparison count exceptions (preserved):")
    for row in short:
        print("  " + ", ".join(row))
    print("No GenWarp, scoring models or VPR experiments executed.")


if __name__ == "__main__":
    main()
