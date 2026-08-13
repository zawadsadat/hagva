import os, csv, shutil, subprocess, sys, re
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
synthesized_dir = "/mnt/sda/Zawad/genwarp/garden_400_out_low"
selected_log    = "/mnt/sda/Zawad/genwarp/garden_400/selected_images.txt"
combined_csv    = "/mnt/sda/Zawad/genwarp/hagva_module_a/gardenspoint_400/validation/combined_summary.csv"
gate_dir        = "/tmp/vprgit_netvlad_gp400_" + os.environ.get("HAGVA_SCORE_COL","mean_combined_score").replace("mean_","").replace("_score","") + "_results"
VPR_DIR         = "/mnt/sda/Zawad/genwarp/VPR-GIT"
STAGING_DIR     = "/tmp/vprgit_netvlad_gp400_stage"
STAGING_QUERY   = f"{STAGING_DIR}/images/GardensPoint/night_right"
STAGING_REF     = f"{STAGING_DIR}/images/GardensPoint/day_right"
REAL_QUERY      = f"{VPR_DIR}/images/GardensPoint/night_right"
REAL_REF        = f"{VPR_DIR}/images/GardensPoint/day_right"
BASELINE_AUC    = 0.199
DESCRIPTOR      = "NetVLAD"
DATASET         = "GardensPoint"
import os as _os
_th_str = _os.environ.get('HAGVA_THRESHOLDS', '0.40')
THRESHOLDS      = [float(x) for x in _th_str.split(',')]
print(f"THRESHOLDS={THRESHOLDS}")
LOADER_PATH     = f"{VPR_DIR}/datasets/load_dataset.py"
PATCH_MARKER    = "# HAGVA_GT_PATCH"
os.makedirs(gate_dir, exist_ok=True)
os.makedirs("/tmp/hagva_vpr_out", exist_ok=True)

# ── Build GT map from selected_images.txt ─────────────────────────────────────
# Format: "000_Image127.jpg" -> injection index 0 -> source Image127 -> ref 127
# GardensPoint identity mapping: ImageXXX -> ref index XXX
selected_gt = {}  # injection_index -> ref_index
with open(selected_log) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('_', 1)          # ["000", "Image127.jpg"]
        idx   = int(parts[0])
        img_name = os.path.splitext(parts[1])[0]   # "Image127"
        img_num  = int(re.search(r'Image(\d+)', img_name).group(1))
        selected_gt[idx] = img_num
print(f"Loaded GT map: {len(selected_gt)} entries")
print(f"Sample: idx=0 -> ref={selected_gt[0]}, idx=1 -> ref={selected_gt[1]}")

# ── Parse synth filename ──────────────────────────────────────────────────────
# Format: "400_000_Image127_ele4_r0.10.jpg"
# Returns: (ele, radius, csv_stem, inj_idx)
# csv_stem = "000_Image127" (matches combined_summary.csv key)
# inj_idx  = 0             (matches selected_gt key)
def parse_synth_filename(filename):
    stem  = os.path.splitext(filename)[0]
    match = re.search(r'_ele([\.\d]+)_r([\d.]+)$', stem)
    if match is None:
        return None
    ele_deg = float(match.group(1))
    radius  = float(match.group(2))
    base    = stem[:match.start()]   # "400_000_Image127"
    # csv_stem: strip total count prefix only -> "000_Image127"
    csv_stem = re.sub(r'^\d+_', '', base, count=1)
    # inj_idx: second number in prefix -> 0
    idx_match = re.match(r'^\d+_(\d+)_', base)
    inj_idx = int(idx_match.group(1)) if idx_match else None
    return ele_deg, radius, csv_stem, inj_idx

# ── Loader patching ───────────────────────────────────────────────────────────
def patch_loader(gt_additions):
    with open(LOADER_PATH, 'r') as f:
        content = f.read()
    if PATCH_MARKER in content:
        content = re.sub(
            rf'\n        {PATCH_MARKER}.*?{PATCH_MARKER}\n',
            '', content, flags=re.DOTALL
        )
    patch_lines = f"\n        {PATCH_MARKER}\n"
    for q_idx, r_idx in gt_additions.items():
        patch_lines += f"        extra_query_gt[{q_idx}] = {r_idx}\n"
    patch_lines += f"        {PATCH_MARKER}\n"
    content = content.replace(
        "        q_basenames = [os.path.basename(f) for f in fns_q]",
        patch_lines + "        q_basenames = [os.path.basename(f) for f in fns_q]"
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
def setup_staging():
    os.makedirs(STAGING_QUERY, exist_ok=True)
    if not os.path.exists(STAGING_REF):
        os.symlink(os.path.abspath(REAL_REF), STAGING_REF)
    for img in glob(os.path.join(REAL_QUERY, "*.jpg")):
        dst = os.path.join(STAGING_QUERY, os.path.basename(img))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(img), dst)
    n = len(glob(os.path.join(STAGING_QUERY, "*.jpg")))
    print(f"Staging ready: {STAGING_QUERY} ({n} images)")

# ── VPR runner ────────────────────────────────────────────────────────────────
def run_vpr(image_paths, gt_additions=None):
    injected = []
    try:
        for src, dst_name in image_paths:
            dst = os.path.join(STAGING_QUERY, dst_name)
            shutil.copy2(src, dst)
            injected.append(dst)
        if gt_additions:
            patch_loader(gt_additions)
        result = subprocess.run(
            [sys.executable, "demo.py",
             "--descriptor", DESCRIPTOR,
             "--dataset", DATASET,
             "--images_dir", f"{STAGING_DIR}/images/GardensPoint",
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
START_IDX = 200
synth_lookup = {}  # csv_stem -> {path, q_idx, ref_idx, dst_name}
q_idx = START_IDX
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
    ref_idx  = selected_gt[inj_idx]
    dst_name = f"Image{q_idx:03d}.jpg"
    synth_lookup[csv_stem] = {
        'path': p, 'q_idx': q_idx,
        'ref_idx': ref_idx, 'dst_name': dst_name
    }
    q_idx += 1
print(f"Synth lookup: {len(synth_lookup)} images (q_idx {START_IDX}..{q_idx-1})")

setup_staging()

# Verify baseline
print("Verifying baseline...")
baseline_check = run_vpr([], {})
print(f"Baseline check AUC: {baseline_check}")

# Naive AUC
print(f"Running naive AUC ({len(synth_lookup)} views)...")
all_paths = [(v['path'], v['dst_name']) for v in synth_lookup.values()]
all_gt    = {v['q_idx']: v['ref_idx'] for v in synth_lookup.values()}
naive_auc = run_vpr(all_paths, all_gt)
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

    paths  = [(v['path'], v['dst_name']) for v in passing.values()]
    gt_add = {v['q_idx']: v['ref_idx'] for v in passing.values()}
    auc = run_vpr(paths, gt_add)
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
    ax.set_title(f"HAGVA Gate: AUC vs Threshold\n(GardensPoint 400 views, PatchNetVLAD)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(gate_dir, 'gate_auc_curve.png'), dpi=150)
    plt.close(fig)

    recovery = (best['auc'] - naive_auc) / (BASELINE_AUC - naive_auc) * 100 if naive_auc else float('nan')
    print("\n" + "="*52)
    print(f"  HAGVA Gate — GardensPoint 400 views (correct GT)")
    print("="*52)
    print(f"  Baseline AUC   : {BASELINE_AUC:.4f}")
    print(f"  Naive AUC      : {naive_auc:.4f}")
    print(f"  Best HAGVA AUC : {best['auc']:.4f}")
    print(f"  Best threshold : {best['threshold']:.2f}")
    print(f"  Views passed   : {best['n_passed']}")
    print(f"  AUC recovery   : {recovery:.1f}%")
    print("="*52)
