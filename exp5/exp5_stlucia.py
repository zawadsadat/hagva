import os, csv, shutil, subprocess, sys, re
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

synthesized_dir = "/mnt/sda/Zawad/genwarp/st_50_out_low"
sig1_csv        = "/mnt/sda/Zawad/genwarp/hagva_module_a/stlucia/signal1_geometric/signal1_summary.csv"
results_dir     = "/tmp/review_exp5_stlucia_results"
VPR_DIR         = "/mnt/sda/Zawad/genwarp/VPR-GIT"
STAGING_DIR     = "/tmp/review_exp5_stlucia_stage"
STAGING_QUERY   = f"{STAGING_DIR}/images/StLucia_small/180809_1545"
STAGING_GT_DIR  = f"{STAGING_DIR}/images/StLucia_small"
REAL_QUERY      = f"{VPR_DIR}/images/StLucia_small/180809_1545"
REAL_REF        = f"{VPR_DIR}/images/StLucia_small/100909_0845"
REAL_GT         = f"{VPR_DIR}/images/StLucia_small/GT.npz"
BASELINE_AUC    = 0.593812
DESCRIPTOR      = "PatchNetVLAD"
DATASET         = "StLucia"
os.makedirs(results_dir, exist_ok=True)
os.makedirs("/tmp/hagva_vpr_out", exist_ok=True)

# ── Load original GT (200x200) ────────────────────────────────────────────────
_gt_orig = np.load(REAL_GT)
GT_hard_orig = _gt_orig['GThard'].astype(bool)
GT_soft_orig = _gt_orig['GTsoft'].astype(bool)
print(f"Original GT shape: {GT_hard_orig.shape}")

def write_gt(extra_cols):
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

def setup_staging():
    os.makedirs(STAGING_QUERY, exist_ok=True)
    ref_dst = f"{STAGING_GT_DIR}/100909_0845"
    if not os.path.exists(ref_dst):
        os.symlink(os.path.abspath(REAL_REF), ref_dst)
    for img in glob(os.path.join(REAL_QUERY, "*.jpg")):
        dst = os.path.join(STAGING_QUERY, os.path.basename(img))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(img), dst)
    write_gt([])
    n = len(glob(os.path.join(STAGING_QUERY, "*.jpg")))
    print(f"Staging ready: {n} images")

_sorted_q = sorted([f for f in os.listdir(REAL_QUERY) if f.endswith('.jpg')])
stem_to_col = {os.path.splitext(f)[0]: i for i, f in enumerate(_sorted_q)}
print(f"Real query files: {len(stem_to_col)}")

def parse_synth_filename(filename):
    stem  = os.path.splitext(filename)[0]
    match = re.search(r'_ele([\.\d]+)_r([\d.]+)$', stem)
    if match is None:
        return None
    base       = stem[:match.start()]
    image_stem = re.sub(r'^\d+_', '', base, count=1)  # e.g. "image018451"
    return image_stem

def run_vpr_single(img_path, source_stem):
    dst_name = "zzz_inject.jpg"
    dst = os.path.join(STAGING_QUERY, dst_name)
    try:
        shutil.copy2(img_path, dst)
        write_gt([stem_to_col[source_stem]])
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
            print(f"  [rc={result.returncode}] [stderr] {result.stderr[-300:]}")
        return auc
    finally:
        if os.path.exists(dst):
            os.remove(dst)
        write_gt([])

# ── Load hallucination % from Signal 1 ───────────────────────────────────────
halluc_pct = {}
with open(sig1_csv, newline='') as f:
    for row in csv.DictReader(f):
        fname = row['synth_file']
        stem  = parse_synth_filename(fname)
        if stem:
            halluc_pct[stem] = float(row.get('pct_hallucinated', 0))
print(f"Loaded hallucination % for {len(halluc_pct)} images")

# ── Build synth list (all 36) ─────────────────────────────────────────────────
synth_list = []
for p in sorted(glob(os.path.join(synthesized_dir, '*.jpg'))):
    fname = os.path.basename(p)
    if 'r15.00' in fname:
        continue
    stem = parse_synth_filename(fname)
    if stem is None:
        continue
    synth_list.append((fname, p, stem))
print(f"Clean synthesized images: {len(synth_list)}")

setup_staging()

# ── Run per-image injection ───────────────────────────────────────────────────
results = []
for i, (fname, path, stem) in enumerate(synth_list):
    hp = halluc_pct.get(stem, 0.0)
    auc = run_vpr_single(path, stem)
    if auc is None:
        print(f"[{i+1}/{len(synth_list)}] ERROR: {fname}")
        continue
    change = auc - BASELINE_AUC
    results.append({'image': fname, 'halluc_pct': hp,
                    'auc': auc, 'auc_change': change})
    print(f"[{i+1}/{len(synth_list)}] {fname}  (halluc: {hp:.1f}%)  AUC={auc:.6f}  change={change:+.6f}")

out_csv = os.path.join(results_dir, 'exp5_stlucia_results.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image','halluc_pct','auc','auc_change'])
    w.writeheader(); w.writerows(results)

halluc_vals = [r['halluc_pct'] for r in results]
auc_changes = [r['auc_change'] for r in results]
r_val, p_val = stats.pearsonr(halluc_vals, auc_changes)
r_sp, p_sp   = stats.spearmanr(halluc_vals, auc_changes)

print("\n" + "="*52)
print(f"  StLucia Exp 5 Results")
print("="*52)
print(f"  Images analysed : {len(results)}")
print(f"  Baseline AUC    : {BASELINE_AUC:.4f}")
print(f"  Pearson  r = {r_val:.4f}  (p={p_val:.4f})")
print(f"  Spearman r = {r_sp:.4f}  (p={p_sp:.4f})")
print("="*52)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(halluc_vals, auc_changes, alpha=0.6, color='steelblue', s=40)
m, b = np.polyfit(halluc_vals, auc_changes, 1)
x_line = np.linspace(min(halluc_vals), max(halluc_vals), 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=1.5,
        label=f'r={r_val:.3f}, p={p_val:.3f}')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Hallucination Percentage (%)", fontsize=11)
ax.set_ylabel(f"AUC Change (baseline = {BASELINE_AUC})", fontsize=11)
ax.set_title("Exp 5: Per-Image AUC Change vs Hallucination %\n(StLucia, PatchNetVLAD)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'exp5_stlucia_plot.png'), dpi=150)
plt.close(fig)
print(f"Saved -> {results_dir}")
