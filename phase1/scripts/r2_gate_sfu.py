import os, csv, shutil, subprocess, sys, re
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
synthesized_dir = "/mnt/sda/Zawad/genwarp/sfu_400_out_low"
selected_log    = "/mnt/sda/Zawad/genwarp/sfu_400/selected_images.txt"
combined_csv    = "/mnt/sda/Zawad/genwarp/hagva_module_a/sfu_400/validation/combined_summary.csv"
combined_csv    = os.environ.get("HAGVA_SCORES_CSV", combined_csv)  # R2 override: extended score CSV
gate_dir        = "/tmp/ablate_sfu400_" + os.environ.get("HAGVA_SCORE_COL","mean_combined_score").replace("mean_","").replace("_score","") + "_results"
VPR_DIR         = "/mnt/sda/Zawad/genwarp/VPR-GIT"
STAGING_DIR     = "/tmp/ablate_sfu400_stage"
STAGING_QUERY   = f"{STAGING_DIR}/images/SFU/jan"
STAGING_REF     = f"{STAGING_DIR}/images/SFU/dry"
STAGING_GT      = f"{STAGING_DIR}/images/SFU"
REAL_QUERY      = f"{VPR_DIR}/images/SFU/jan"
REAL_REF        = f"{VPR_DIR}/images/SFU/dry"
REAL_GT_DIR     = "/mnt/sda/Zawad/genwarp/VPR-Query-NEW/images/SFU"
LOADER_PATH     = f"{VPR_DIR}/datasets/load_dataset.py"
PATCH_MARKER    = "# HAGVA_SFU400_GT_PATCH"
BASELINE_AUC    = 0.729
DESCRIPTOR      = "PatchNetVLAD"
DATASET         = "SFU"
import os as _os
_th_str = _os.environ.get('HAGVA_THRESHOLDS', '0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50')
THRESHOLDS      = [float(x) for x in _th_str.split(',')]
print(f"THRESHOLDS={THRESHOLDS}")
os.makedirs(gate_dir, exist_ok=True)
os.makedirs("/tmp/hagva_vpr_out", exist_ok=True)

# ── Build GT map from selected_images.txt ─────────────────────────────────────
# Format: "000_b008.jpg" -> injection index 0 -> source "b008.jpg"
selected_gt = {}  # injection_index -> source_filename (e.g. "b008.jpg")
with open(selected_log) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('_', 1)       # ["000", "b008.jpg"]
        idx  = int(parts[0])
        src_file = parts[1]              # "b008.jpg"
        selected_gt[idx] = src_file
print(f"Loaded GT map: {len(selected_gt)} entries")
print(f"Sample: idx=0 -> {selected_gt[0]}, idx=1 -> {selected_gt[1]}")

# ── Parse synth filename ──────────────────────────────────────────────────────
# Format: "001_000_b008_ele1_r0.10.jpg"
# Returns: (ele, radius, csv_stem, inj_idx)
def parse_synth_filename(filename):
    stem  = os.path.splitext(filename)[0]
    match = re.search(r'_ele([\.\d]+)_r([\d.]+)$', stem)
    if match is None:
        return None
    ele_deg = float(match.group(1))
    radius  = float(match.group(2))
    base    = stem[:match.start()]       # "001_000_b008"
    csv_stem  = re.sub(r'^\d+_', '', base, count=1)   # "000_b008"
    idx_match = re.match(r'^\d+_(\d+)_', base)
    inj_idx   = int(idx_match.group(1)) if idx_match else None
    return ele_deg, radius, csv_stem, inj_idx

# ── Loader patching ───────────────────────────────────────────────────────────
def patch_loader(copy_from_additions):
    with open(LOADER_PATH, 'r') as f:
        content = f.read()
    if PATCH_MARKER in content:
        content = re.sub(
            rf'\n        {PATCH_MARKER}.*?{PATCH_MARKER}\n',
            '', content, flags=re.DOTALL
        )
    patch = f"\n        {PATCH_MARKER}\n"
    for dst, src in copy_from_additions.items():
        patch += f"        copy_from['{dst}'] = '{src}'\n"
    patch += f"        {PATCH_MARKER}\n"
    # Insert after copy_from dict definition closing brace
    content = content.replace(
        "        # Resize hard GT to match current number of queries (expand or truncate)",
        patch + "        # Resize hard GT to match current number of queries (expand or truncate)"
    )
    with open(LOADER_PATH, 'w') as f:
        f.write(content)

def unpatch_loader():
    with open(LOADER_PATH, 'r') as f:
        content = f.read()
    if PATCH_MARKER in content:
        content = re.sub(
            rf'\n        {PATCH_MARKER}.*?{PATCH_MARKER}\n',
            '', content, flags=re.DOTALL
        )
    with open(LOADER_PATH, 'w') as f:
        f.write(content)

# ── Staging setup ─────────────────────────────────────────────────────────────
def restore_gt():
    # Always restore from GT_original to prevent accumulation
    shutil.copy2(
        os.path.join(REAL_GT_DIR, "GT_original.npz"),
        os.path.join(STAGING_GT, "GT.npz")
    )
    # Remove GT_original.npz from staging if it exists to prevent loader confusion
    gt_orig_staging = os.path.join(STAGING_GT, "GT_original.npz")
    if os.path.exists(gt_orig_staging):
        os.remove(gt_orig_staging)

def setup_staging():
    os.makedirs(STAGING_QUERY, exist_ok=True)
    if not os.path.exists(STAGING_REF):
        os.symlink(os.path.abspath(REAL_REF), STAGING_REF)
    restore_gt()
    # Symlink only original 385 query images
    for img in sorted(glob(os.path.join(REAL_QUERY, "*.jpg")))[:385]:
        dst = os.path.join(STAGING_QUERY, os.path.basename(img))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(img), dst)
    n = len(glob(os.path.join(STAGING_QUERY, "*.jpg")))
    print(f"Staging ready: {STAGING_QUERY} ({n} images)")

# ── VPR runner ────────────────────────────────────────────────────────────────
def run_vpr(image_items):
    # Remove expanded GT from staging before restore
    gt_staging = os.path.join(STAGING_GT, "GT.npz")
    if os.path.exists(gt_staging):
        os.remove(gt_staging)
    restore_gt()
    injected = []
    copy_from_additions = {}
    try:
        for src_path, dst_name, source_file in image_items:
            dst = os.path.join(STAGING_QUERY, dst_name)
            shutil.copy2(src_path, dst)
            injected.append(dst)
            copy_from_additions[dst_name] = source_file
        if copy_from_additions:
            patch_loader(copy_from_additions)
        result = subprocess.run(
            [sys.executable, "demo.py",
             "--descriptor", DESCRIPTOR,
             "--dataset", DATASET,
             "--images_dir", STAGING_GT,
             "--output_dir", "/tmp/hagva_vpr_out"],
            capture_output=True, text=True,
            cwd=VPR_DIR,
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
        auc = None
        for line in result.stdout.splitlines():
            if "AUC (area under curve)" in line:
                try:
                    auc = float(line.strip().split(":")[-1].strip())
                except ValueError:
                    pass
                break
        if auc is None:
            print(f"  [rc={result.returncode}]")
            print(f"  [stderr] {result.stderr[-400:]}")
        return auc
    finally:
        for dst in injected:
            try: os.remove(dst)
            except: pass
        unpatch_loader()

# ── Load combined scores ──────────────────────────────────────────────────────
combined_scores = {}
with open(combined_csv, newline='') as f:
    for row in csv.DictReader(f):
        combined_scores[row['image_stem']] = float(row[os.environ.get('HAGVA_SCORE_COL','mean_combined_score')])
print(f"Loaded {len(combined_scores)} combined scores")

# ── Build synth lookup ────────────────────────────────────────────────────────
synth_lookup = {}  # csv_stem -> {path, dst_name, source_file}
for p in sorted(glob(os.path.join(synthesized_dir, '*.jpg'))):
    fname = os.path.basename(p)
    if 'r15.00' in fname:
        continue
    parsed = parse_synth_filename(fname)
    if parsed is None:
        continue
    _, _, csv_stem, inj_idx = parsed
    if inj_idx is None or inj_idx not in selected_gt:
        print(f"  SKIP (no GT): {fname}")
        continue
    source_file = selected_gt[inj_idx]   # e.g. "b008.jpg"
    dst_name    = f"z_hagva_tmp_{fname}"
    synth_lookup[csv_stem] = {
        'path': p,
        'dst_name': dst_name,
        'source_file': source_file
    }
print(f"Synth lookup: {len(synth_lookup)} images")

setup_staging()

# Verify baseline
print("Verifying baseline...")
baseline_check = run_vpr([])
print(f"Baseline check AUC: {baseline_check}")

# Naive AUC
print(f"Running naive AUC ({len(synth_lookup)} views)...")
all_items = [(v['path'], v['dst_name'], v['source_file'])
             for v in synth_lookup.values()]
naive_auc = run_vpr(all_items)
print(f"Naive AUC: {naive_auc}")

# Threshold sweep
gate_results = []
for thresh in THRESHOLDS:
    passing = {s: v for s, v in synth_lookup.items()
               if s in combined_scores and combined_scores[s] < thresh}
    n_passed   = len(passing)
    n_rejected = len(synth_lookup) - n_passed

    if n_passed == 0:
        auc = BASELINE_AUC
        dn  = auc - naive_auc if naive_auc else float('nan')
        gate_results.append({'threshold': thresh, 'n_passed': 0,
            'n_rejected': n_rejected, 'auc': auc,
            'delta_from_baseline': 0.0, 'delta_from_naive': dn})
        print(f"  thresh={thresh:.2f}  passed=  0  rejected={n_rejected:3d}  AUC={auc:.4f} (zero-pass)")
        continue

    items = [(v['path'], v['dst_name'], v['source_file'])
             for v in passing.values()]
    auc = run_vpr(items)
    if auc is None:
        print(f"  thresh={thresh:.2f}  ERROR"); continue

    db = auc - BASELINE_AUC
    dn = auc - naive_auc if naive_auc else float('nan')
    gate_results.append({'threshold': thresh, 'n_passed': n_passed,
        'n_rejected': n_rejected, 'auc': auc,
        'delta_from_baseline': db, 'delta_from_naive': dn})
    print(f"  thresh={thresh:.2f}  passed={n_passed:3d}  rejected={n_rejected:3d}"
          f"  AUC={auc:.4f}  Δbase={db:+.4f}  Δnaive={dn:+.4f}")

# Save CSV
gate_csv = os.path.join(gate_dir, 'gate_results.csv')
with open(gate_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['threshold','n_passed','n_rejected',
                                       'auc','delta_from_baseline','delta_from_naive'])
    w.writeheader(); w.writerows(gate_results)
print(f"Saved -> {gate_csv}")

# Plot
valid = [r for r in gate_results if r['n_passed'] > 0]
if valid:
    best = max(valid, key=lambda r: r['auc'])
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axhline(BASELINE_AUC, color='green', linestyle='--', linewidth=1.5,
               label=f"No injection baseline ({BASELINE_AUC:.3f})")
    if naive_auc:
        ax.axhline(naive_auc, color='red', linestyle='--', linewidth=1.5,
                   label=f"Naive injection ({naive_auc:.3f})")
    zt = [r['threshold'] for r in gate_results if r['n_passed']==0]
    za = [r['auc'] for r in gate_results if r['n_passed']==0]
    nt = [r['threshold'] for r in gate_results if r['n_passed']>0]
    nv = [r['auc'] for r in gate_results if r['n_passed']>0]
    ax.plot(zt, za, 'o', color='lightblue', markersize=7,
            label="Zero views passed (= baseline)")
    ax.plot(nt, nv, 'b-o', linewidth=2, markersize=6, label="HAGVA gate AUC")
    ax.plot(best['threshold'], best['auc'], 'b*', markersize=14,
            label=f"Best: thresh={best['threshold']:.2f}, AUC={best['auc']:.4f}")
    ax.set_xlabel("Rejection Threshold", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"HAGVA Gate: AUC vs Threshold\n(SFU 400 views, PatchNetVLAD)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(gate_dir, 'gate_auc_curve.png'), dpi=150)
    plt.close(fig)

    recovery = (best['auc'] - naive_auc) / (BASELINE_AUC - naive_auc) * 100 if naive_auc else float('nan')
    print("\n" + "="*52)
    print(f"  HAGVA Gate — SFU 400 views (correct GT)")
    print("="*52)
    print(f"  Baseline AUC   : {BASELINE_AUC:.4f}")
    print(f"  Naive AUC      : {naive_auc:.4f}")
    print(f"  Best HAGVA AUC : {best['auc']:.4f}")
    print(f"  Best threshold : {best['threshold']:.2f}")
    print(f"  Views passed   : {best['n_passed']}")
    print(f"  AUC recovery   : {recovery:.1f}%")
    print("="*52)
