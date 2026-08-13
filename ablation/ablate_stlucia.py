import os, csv, shutil, subprocess, sys, re
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

synthesized_dir = "/mnt/sda/Zawad/genwarp/st_400_out_low"
selected_log    = "/mnt/sda/Zawad/genwarp/st_400/selected_images.txt"
combined_csv    = "/mnt/sda/Zawad/genwarp/hagva_module_a/stlucia_400/validation/combined_summary.csv"
gate_dir        = "/tmp/ablate_stlucia400_" + os.environ.get("HAGVA_SCORE_COL","mean_combined_score").replace("mean_","").replace("_score","") + "_results"
VPR_DIR         = "/mnt/sda/Zawad/genwarp/VPR-GIT"
STAGING_DIR     = "/tmp/ablate_stlucia400_stage"
STAGING_QUERY   = f"{STAGING_DIR}/images/StLucia_small/180809_1545"
STAGING_REF     = f"{STAGING_DIR}/images/StLucia_small/100909_0845"
STAGING_GT_DIR  = f"{STAGING_DIR}/images/StLucia_small"
REAL_QUERY      = f"{VPR_DIR}/images/StLucia_small/180809_1545"
REAL_REF        = f"{VPR_DIR}/images/StLucia_small/100909_0845"
REAL_GT         = f"{VPR_DIR}/images/StLucia_small/GT.npz"
BASELINE_AUC    = 0.594
DESCRIPTOR      = "PatchNetVLAD"
DATASET         = "StLucia"
import os as _os
_th_str = _os.environ.get('HAGVA_THRESHOLDS', '0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50')
THRESHOLDS      = [float(x) for x in _th_str.split(',')]

# ── Load original GT (200x200) ────────────────────────────────────────────────
_gt_orig = np.load(REAL_GT)
GT_hard_orig = _gt_orig['GThard'].astype(bool)
GT_soft_orig = _gt_orig['GTsoft'].astype(bool)
print(f"Original GT shape: {GT_hard_orig.shape}")

def write_gt(extra_cols):
    """extra_cols: list of source column indices (0-199) for injected queries"""
    if len(extra_cols) == 0:
        np.savez(os.path.join(STAGING_GT_DIR, "GT.npz"),
                 GThard=GT_hard_orig, GTsoft=GT_soft_orig)
        return
    extra_hard = GT_hard_orig[:, extra_cols]
    extra_soft = GT_soft_orig[:, extra_cols]
    GT_hard_ext = np.hstack([GT_hard_orig, extra_hard])
    GT_soft_ext = np.hstack([GT_soft_orig, extra_soft])
    np.savez(os.path.join(STAGING_GT_DIR, "GT.npz"),
             GThard=GT_hard_ext, GTsoft=GT_soft_ext)
os.makedirs(gate_dir, exist_ok=True)
os.makedirs("/tmp/hagva_vpr_out", exist_ok=True)

# ── GT map from selected_images.txt ──────────────────────────────────────────
# Format: "000_image020341.jpg" -> index 0 -> source stem "image020341"
selected_gt = {}
with open(selected_log) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('_', 1)
        idx = int(parts[0])
        src_stem = os.path.splitext(parts[1])[0]
        selected_gt[idx] = src_stem
print(f"Loaded GT map: {len(selected_gt)} entries")
print(f"Sample: idx=0 -> {selected_gt[0]}, idx=1 -> {selected_gt[1]}")

# ── Parse synth filename ──────────────────────────────────────────────────────
def parse_synth_filename(filename):
    stem  = os.path.splitext(filename)[0]
    match = re.search(r'_ele([\.\d]+)_r([\d.]+)$', stem)
    if match is None:
        return None
    base      = stem[:match.start()]
    csv_stem  = re.sub(r'^\d+_', '', base, count=1)
    idx_match = re.match(r'^\d+_(\d+)_', base)
    inj_idx   = int(idx_match.group(1)) if idx_match else None
    return float(match.group(1)), float(match.group(2)), csv_stem, inj_idx

# ── Find query stem -> column index mapping (sorted order = GT column order) ──
_sorted_q = sorted([f for f in os.listdir(REAL_QUERY) if f.endswith('.jpg')])
stem_to_col = {os.path.splitext(f)[0]: i for i, f in enumerate(_sorted_q)}
print(f"Real query files: {len(stem_to_col)}")

# ── Staging setup ─────────────────────────────────────────────────────────────
def setup_staging():
    os.makedirs(STAGING_QUERY, exist_ok=True)
    if not os.path.exists(STAGING_REF):
        os.symlink(os.path.abspath(REAL_REF), STAGING_REF)
    write_gt([])
    for img in glob(os.path.join(REAL_QUERY, "*.jpg")):
        dst = os.path.join(STAGING_QUERY, os.path.basename(img))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(img), dst)
    n = len(glob(os.path.join(STAGING_QUERY, "*.jpg")))
    print(f"Staging ready: {STAGING_QUERY} ({n} images)")

# ── VPR runner ────────────────────────────────────────────────────────────────
def run_vpr(image_items):
    """image_items: list of (path, source_stem) tuples"""
    injected = []
    extra_cols = []
    try:
        for i, (src, source_stem) in enumerate(image_items):
            new_name = f"zzz_inject_{i:04d}.jpg"
            dst = os.path.join(STAGING_QUERY, new_name)
            shutil.copy2(src, dst)
            injected.append(dst)
            extra_cols.append(stem_to_col[source_stem])
        write_gt(extra_cols)
        result = subprocess.run(
            [sys.executable, "demo.py",
             "--descriptor", DESCRIPTOR,
             "--dataset", DATASET,
             "--images_dir", STAGING_GT_DIR,
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
        write_gt([])

# ── Load combined scores ──────────────────────────────────────────────────────
combined_scores = {}
with open(combined_csv, newline='') as f:
    for row in csv.DictReader(f):
        combined_scores[row['image_stem']] = float(row[os.environ.get('HAGVA_SCORE_COL','mean_combined_score')])
print(f"Loaded {len(combined_scores)} combined scores")

# ── Build synth lookup ────────────────────────────────────────────────────────
synth_lookup = {}
for p in sorted(glob(os.path.join(synthesized_dir, '*.jpg'))):
    fname = os.path.basename(p)
    if 'r15.00' in fname:
        continue
    parsed = parse_synth_filename(fname)
    if parsed is None:
        continue
    _, _, csv_stem, inj_idx = parsed
    if inj_idx is None or inj_idx not in selected_gt:
        continue
    synth_lookup[csv_stem] = {'path': p, 'source_stem': selected_gt[inj_idx]}
print(f"Synth lookup: {len(synth_lookup)} images")

setup_staging()

print("Verifying baseline...")
baseline_check = run_vpr([])
print(f"Baseline check AUC: {baseline_check}")

print(f"Running naive AUC ({len(synth_lookup)} views)...")
naive_auc = run_vpr([(v['path'], v['source_stem']) for v in synth_lookup.values()])
print(f"Naive AUC: {naive_auc}")

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

    items = [(v['path'], v['source_stem']) for v in passing.values()]
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

gate_csv = os.path.join(gate_dir, 'gate_results.csv')
with open(gate_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['threshold','n_passed','n_rejected',
                                       'auc','delta_from_baseline','delta_from_naive'])
    w.writeheader(); w.writerows(gate_results)
print(f"Saved -> {gate_csv}")

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
    ax.set_title(f"HAGVA Gate: AUC vs Threshold\n(StLucia 400 views, PatchNetVLAD)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(gate_dir, 'gate_auc_curve.png'), dpi=150)
    plt.close(fig)

    recovery = (best['auc'] - naive_auc) / (BASELINE_AUC - naive_auc) * 100 if naive_auc else float('nan')
    print("\n" + "="*52)
    print(f"  HAGVA Gate — StLucia 400 views")
    print("="*52)
    print(f"  Baseline AUC   : {BASELINE_AUC:.4f}")
    print(f"  Naive AUC      : {naive_auc:.4f}")
    print(f"  Best HAGVA AUC : {best['auc']:.4f}")
    print(f"  Best threshold : {best['threshold']:.2f}")
    print(f"  Views passed   : {best['n_passed']}")
    print(f"  AUC recovery   : {recovery:.1f}%")
    print("="*52)
