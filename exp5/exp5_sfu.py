import os, re, csv, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from glob import glob

BASELINE_AUC    = 0.729180
synthesized_dir = "/mnt/sda/Zawad/genwarp/sfu_50_out_low"
signal1_csv     = "/mnt/sda/Zawad/genwarp/hagva_module_a/sfu/signal1_geometric/signal1_summary.csv"
VPR_DIR         = "/mnt/sda/Zawad/genwarp/VPR-GIT"
STAGING_DIR     = "/tmp/review_exp5_sfu_stage"
STAGING_QUERY   = f"{STAGING_DIR}/images/SFU/jan"
STAGING_REF     = f"{STAGING_DIR}/images/SFU/dry"
STAGING_GT      = f"{STAGING_DIR}/images/SFU"
REAL_QUERY      = f"{VPR_DIR}/images/SFU/jan"
REAL_REF        = f"{VPR_DIR}/images/SFU/dry"
REAL_GT         = "/mnt/sda/Zawad/genwarp/VPR-Query-NEW/images/SFU"
LOADER_PATH     = f"{VPR_DIR}/datasets/load_dataset.py"
PATCH_MARKER    = "# HAGVA_GT_PATCH"
ANCHOR          = "        # Resize hard GT to match current number of queries (expand or truncate)"
DESCRIPTOR      = "PatchNetVLAD"
DATASET         = "SFU"
out_dir         = "/tmp/review_exp5_sfu_results"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("/tmp/hagva_vpr_out", exist_ok=True)

def unpatch_loader():
    c = open(LOADER_PATH).read()
    c = re.sub(rf'\n        {PATCH_MARKER}.*?{PATCH_MARKER}\n', '', c, flags=re.DOTALL)
    open(LOADER_PATH, 'w').write(c)

def patch_loader(dst_name, src_name):
    unpatch_loader()
    c = open(LOADER_PATH).read()
    if ANCHOR not in c:
        raise RuntimeError("anchor not found in loader")
    block = (f"\n        {PATCH_MARKER}\n"
             f"        copy_from['{dst_name}'] = '{src_name}'\n"
             f"        {PATCH_MARKER}\n")
    open(LOADER_PATH, 'w').write(c.replace(ANCHOR, block + ANCHOR))
    if f"copy_from['{dst_name}']" not in open(LOADER_PATH).read():
        raise RuntimeError("patch verification failed")

def restore_gt():
    shutil.copy2(os.path.join(REAL_GT, "GT.npz"), os.path.join(STAGING_GT, "GT.npz"))
    shutil.copy2(os.path.join(REAL_GT, "GT.npz"), os.path.join(STAGING_GT, "GT_original.npz"))

def setup_staging():
    os.makedirs(STAGING_QUERY, exist_ok=True)
    if not os.path.lexists(STAGING_REF):
        os.symlink(os.path.abspath(REAL_REF), STAGING_REF)
    restore_gt()
    for img in glob(os.path.join(REAL_QUERY, "*.jpg")):
        dst = os.path.join(STAGING_QUERY, os.path.basename(img))
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(img), dst)
    print(f"Staging ready: {len(glob(os.path.join(STAGING_QUERY,'*.jpg')))} images")

def run_single(img_path, src_name):
    dst_name = f"z_hagva_tmp_{os.path.basename(img_path)}"
    dst = os.path.join(STAGING_QUERY, dst_name)
    try:
        restore_gt()
        shutil.copy2(img_path, dst)
        patch_loader(dst_name, src_name)
        r = subprocess.run(
            [sys.executable, "demo.py", "--descriptor", DESCRIPTOR,
             "--dataset", DATASET, "--images_dir", STAGING_GT,
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
        restore_gt()

halluc, srcmap = {}, {}
with open(signal1_csv, newline='') as f:
    for row in csv.DictReader(f):
        stem = re.sub(r'^\d+_', '', os.path.splitext(row['synth_file'])[0])
        stem = re.sub(r'_ele[\d.]+_r[\d.]+$', '', stem)
        halluc[stem] = float(row['pct_hallucinated'])
        srcmap[stem] = row['source_file']
print(f"Loaded {len(halluc)} entries from signal1_summary")

items = []
for p in sorted(glob(os.path.join(synthesized_dir, '*.jpg'))):
    fn = os.path.basename(p)
    if 'r15.00' in fn: continue
    stem = re.sub(r'^\d+_', '', os.path.splitext(fn)[0])
    stem = re.sub(r'_ele[\d.]+_r[\d.]+$', '', stem)
    if stem not in halluc:
        print(f"  SKIP (no signal1 entry): {fn}"); continue
    items.append((fn, p, stem, srcmap[stem]))
print(f"Views to process: {len(items)}")

setup_staging()
unpatch_loader()

results = []
for i, (fn, path, stem, src_name) in enumerate(items, 1):
    auc = run_single(path, src_name)
    if auc is None:
        print(f"[{i}/{len(items)}] ERROR {fn}"); continue
    ch = auc - BASELINE_AUC
    results.append({'image': fn, 'halluc_pct': halluc[stem], 'auc': auc, 'auc_change': ch})
    print(f"[{i}/{len(items)}] {fn}  src={src_name}  (halluc {halluc[stem]:.1f}%)  AUC={auc:.6f}  change={ch:+.6f}")

with open(os.path.join(out_dir,'exp5_sfu_results.csv'),'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image','halluc_pct','auc','auc_change'])
    w.writeheader(); w.writerows(results)

h = [r['halluc_pct'] for r in results]; c = [r['auc_change'] for r in results]
rv, pv = stats.pearsonr(h, c); rs, ps = stats.spearmanr(h, c)
print("\n"+"="*52)
print("  SFU Exp 5 (corrected GT)")
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
ax.set_title("Exp 5: SFU (PatchNetVLAD)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(out_dir,'exp5_sfu_plot.png'), dpi=150)
print(f"Saved -> {out_dir}")
