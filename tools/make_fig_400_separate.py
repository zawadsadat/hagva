import os, re
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

val_dir   = "/mnt/sda/Zawad/genwarp/hagva_module_a/gardenspoint_400/validation"
synth_dir = "/mnt/sda/Zawad/genwarp/garden_400_out_low"
src_dir   = "/mnt/sda/Zawad/genwarp/garden_400"
out_dir   = "/tmp/hagva_figures"
os.makedirs(out_dir, exist_ok=True)

def score_to_colormap(score_map):
    colored = mcm.RdYlBu_r(score_map)
    return (colored[..., :3] * 255).astype(np.uint8)

def find_synth(csv_stem, synth_dir):
    idx, img_name = csv_stem.split('_', 1)
    for f in os.listdir(synth_dir):
        if f.endswith('.jpg') and 'r15.00' not in f:
            m = re.match(r'^\d+_(\d+)_(.+)_ele', f)
            if m and m.group(1) == idx and m.group(2) == img_name:
                return os.path.join(synth_dir, f)
    return None

def find_src(csv_stem, src_dir):
    img_name = csv_stem.split('_', 1)[1]
    for f in os.listdir(src_dir):
        if img_name in f:
            return os.path.join(src_dir, f)
    return None

cases = [
    ("103_Image107", "ACCEPTED", r"score = 0.3838 < $\tau$ = 0.40", "green", "fig_comparison_Image107"),
    ("318_Image030", "REJECTED", r"score = 0.5420 $\geq$ $\tau$ = 0.40", "red", "fig_comparison_Image030"),
]

for csv_stem, label, score_str, color, out_name in cases:
    src_path   = find_src(csv_stem, src_dir)
    synth_path = find_synth(csv_stem, synth_dir)
    comb_path  = os.path.join(val_dir, f"{csv_stem}_combined_score.npy")

    if src_path is None or synth_path is None or not os.path.exists(comb_path):
        print(f"MISSING for {csv_stem}: src={src_path}, synth={synth_path}, comb_exists={os.path.exists(comb_path)}")
        continue

    src_img   = Image.open(src_path).convert('RGB')
    synth_img = Image.open(synth_path).convert('RGB')
    score_img = Image.fromarray(score_to_colormap(np.load(comb_path)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2),
                             gridspec_kw={'wspace': 0.02})
    fig.patch.set_facecolor('white')

    titles = ["Source Image", "Synthesized (GenWarp)", "Combined Hallucination Score"]
    for col, img in enumerate([src_img, synth_img, score_img]):
        ax = axes[col]
        ax.set_facecolor('white')
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(titles[col], fontsize=11, pad=5)

    axes[0].text(
        -0.02, 0.5, f"{label}\n{score_str}",
        transform=axes[0].transAxes,
        fontsize=10, fontweight='bold', color=color,
        ha='right', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=color, linewidth=2)
    )

    out_path = os.path.join(out_dir, f"{out_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved -> {out_path}")
