import os, csv, re, shutil, subprocess, sys
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

synthesized_dir = "/mnt/sda/Zawad/genwarp/garden_50_out_low"
sig1_csv        = "/mnt/sda/Zawad/genwarp/hagva_module_a/gardenspoint/signal1_geometric/signal1_summary.csv"
results_dir     = "/tmp/review_exp5_gp_results"
VPR_DIR         = "/mnt/sda/Zawad/genwarp/VPR-GIT"
STAGING_DIR     = "/tmp/review_exp5_gp_stage"
STAGING_QUERY   = f"{STAGING_DIR}/images/GardensPoint/night_right"
STAGING_REF     = f"{STAGING_DIR}/images/GardensPoint/day_right"
REAL_QUERY      = f"{VPR_DIR}/images/GardensPoint/night_right"
REAL_REF        = f"{VPR_DIR}/images/GardensPoint/day_right"
LOADER_PATH     = f"{VPR_DIR}/datasets/load_dataset.py"
PATCH_MARKER    = "# HAGVA_GT_PATCH"
ANCHOR          = "        q_basenames = [os.path.basename(f) for f in fns_q]"
BASELINE_AUC    = 0.801858
DESCRIPTOR      = "PatchNetVLAD"
DATASET         = "GardensPoint"
INJECT_IDX      = 200
os.makedirs(results_dir, exist_ok=True)
os.makedirs("/tmp/hagva_vpr_out", exist_ok=True)

def unpatch_loader():
    c = open(LOADER_PATH).read()
    c = re.sub(rf'\n        {PATCH_MARKER}.*?{PATCH_MARKER}\n', '', c, flags=re.DOTALL)
    open(LOADER_PATH, 'w').write(c)

def patch_loader(q_idx, ref_idx):
    unpatch_loader()
    c = open(LOADER_PATH).read()
    if ANCHOR not in c:
        raise RuntimeError("anchor line not found in loader")
    block = (f"\n        {PATCH_MARKER}\n"
             f"        extra_query_gt[{q_idx}] = {ref_idx}\n"
             f"        {PATCH_MARKER}\n")
    open(LOADER_PATH, 'w').write(c.replace(ANCHOR, block + ANCHOR))
    if f"extra_query_gt[{q_idx}]" not in open(LOADER_PATH).read():
        raise RuntimeError("patch verification failed")

def setup_staging():
    os.makedirs(STAGING_QUERY, exist_ok=True)
    if not os.path.lexists(STAGING_REF):
        os.symlink(os.path.abspath(REAL_REF), STAGING_REF)
    for img in glob(os.path.join(REAL_QUERY, "*.jpg")):
        dst = os.path.join(STAGING_QUERY, os.path.basename(img))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(img), dst)
    print(f"Staging ready: {len(glob(os.path.join(STAGING_QUERY,'*.jpg')))} images")

def run_single(img_path, ref_idx):
    dst = os.path.join(STAGING_QUERY, f"Image{INJECT_IDX:03d}.jpg")
    try:
        shutil.copy2(img_path, dst)
        patch_loader(INJECT_IDX, ref_idx)
        r = subprocess.run(
            [sys.executable, "demo.py", "--descriptor", DESCRIPTOR,
             "--dataset", DATASET,
             "--images_dir", f"{STAGING_DIR}/images/GardensPoint",
             "--output_dir", "/tmp/hagva_vpr_out"],
            capture_output=True, text=True, cwd=VPR_DIR,
            env={**os.environ, "MPLBACKEND": "Agg"})
        for line in r.stdout.splitlines():
            if "AUC (area under curve)" in line:
                try: return float(line.split(":")[-1].strip())
                except ValueError: pass
        print(f"  [rc={r.returncode}] {r.stderr[-300:]}")
        return None
    finally:
        if os.path.exists(dst): os.remove(dst)
        unpatch_loader()

halluc, srcmap = {}, {}
with open(sig1_csv, newline='') as f:
    for row in csv.DictReader(f):
        stem = re.sub(r'^\d+_', '', os.path.splitext(row['synth_file'])[0])
        halluc[stem] = float(row['pct_hallucinated'])
        srcmap[stem] = int(re.search(r'Image(\d+)', row['source_file']).group(1))
print(f"Loaded {len(halluc)} entries from signal1_summary")

items = []
for p in sorted(glob(os.path.join(synthesized_dir, '*.jpg'))):
    fn = os.path.basename(p)
    if 'r15.00' in fn: continue
    stem = re.sub(r'^\d+_', '', os.path.splitext(fn)[0])
    if stem not in halluc:
        print(f"  SKIP (no signal1 entry): {fn}"); continue
    items.append((fn, p, stem, srcmap[stem]))
print(f"Views to process: {len(items)}")

setup_staging()
unpatch_loader()

results = []
for i, (fn, path, stem, ref_idx) in enumerate(items, 1):
    auc = run_single(path, ref_idx)
    if auc is None:
        print(f"[{i}/{len(items)}] ERROR {fn}"); continue
    ch = auc - BASELINE_AUC
    results.append({'image': fn, 'halluc_pct': halluc[stem], 'auc': auc, 'auc_change': ch})
    print(f"[{i}/{len(items)}] {fn}  ref={ref_idx}  (halluc {halluc[stem]:.1f}%)  AUC={auc:.6f}  change={ch:+.6f}")

with open(os.path.join(results_dir,'exp5_gp_results.csv'),'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image','halluc_pct','auc','auc_change'])
    w.writeheader(); w.writerows(results)

h = [r['halluc_pct'] for r in results]; c = [r['auc_change'] for r in results]
rv, pv = stats.pearsonr(h, c); rs, ps = stats.spearmanr(h, c)
print("\n"+"="*52)
print("  GardensPoint Exp 5 (corrected GT)")
print("="*52)
print(f"  Images   : {len(results)}")
print(f"  Baseline : {BASELINE_AUC:.4f}")
print(f"  Pearson  r = {rv:.4f}  (p={pv:.4f})")
print(f"  Spearman r = {rs:.4f}  (p={ps:.4f})")
print("="*52)

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(h, c, alpha=.6, color='steelblue', s=40)
m,b = np.polyfit(h,c,1); xs = np.linspace(min(h),max(h),100)
ax.plot(xs, m*xs+b, 'r-', lw=1.5, label=f'r={rv:.3f}, p={pv:.3f}')
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel("Hallucination Percentage (%)"); ax.set_ylabel(f"AUC Change (baseline = {BASELINE_AUC})")
ax.set_title("Exp 5: GardensPoint (PatchNetVLAD)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(results_dir,'exp5_gp_plot.png'), dpi=150)
print(f"Saved -> {results_dir}")
