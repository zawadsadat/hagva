#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 3b: score a synthesized-view pool with Module A's Signal 2 (semantic drift) and the corrected Signal 1.

Signal 2 = exact re-implementation of module_a/VPR.ipynb cell 3 (+ cell 8):
    DINOv2 ViT-S/14 (torch.hub dinov2_vits14), T.Resize((448,448)) -> ToTensor -> ImageNet Normalize,
    get_intermediate_layers(x, n=1, return_class_token=False)[0] -> (1, 1024, 384) patch tokens -> (32, 32, 384),
    cos_dist = clamp(1 - cosine(src, synth), 0, 2) per patch, bilinear upsample (align_corners=False) to the SOURCE
    image size (cell 3 line 131: orig_w, orig_h = src_pil.size), score = clip(dist / p99(dist), 0, 1),
    mean_sem_score = score.mean() written with 4 decimals, as Module A does.
Signal 1 (corrected) = phase2/scripts/recompute_geo.py formula (Module A cell 2 compute_geometric_score verbatim) on the
    regenerated warp-only render and mask, both resized to the synthesized view (BILINEAR image, NEAREST mask).
holefrac = fraction of mask pixels < 0.5 after /255 (PNG value < 128) at the synthesized size.

  score_pool.py --synth-dir D --source-dir D --warp-dir D --mask-dir D --out CSV [--device cuda]
Output columns: image_stem, synth_file, source_file, ele, r, holefrac, mean_sem_score, geo_fixed, mean_combined_fixed
"""
import argparse, csv, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

DINOV2_COMMIT = "85a24602099d397264d5b30461ad7f3bfd726ca1"   # same dinov2_vits14_pretrain.pth weights as hub 'main' (python-3.9-compatible snapshot)

def parse(fname):
    stem = os.path.splitext(fname)[0]; m = re.search(r'_ele([\d.]+)_r([\d.]+)$', stem)
    base = stem[:m.start()]; return re.sub(r'^\d+_', '', base, count=1), float(m.group(1)), float(m.group(2))

def extract_dino_features(pil_img, dino, transform, device):          # cell 3 verbatim
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino.get_intermediate_layers(x, n=1, return_class_token=False)[0]
    feat_dim = feats.shape[-1]; num_patches = feats.shape[1]; h = w = int(num_patches ** 0.5)
    return feats[0].reshape(h, w, feat_dim).cpu().float()

def compute_semantic_score(src_pil, synth_pil, dino, transform, device, orig_h, orig_w):   # cell 3 verbatim
    src_feat = extract_dino_features(src_pil, dino, transform, device); synth_feat = extract_dino_features(synth_pil, dino, transform, device)
    cos_sim = F.cosine_similarity(src_feat.unsqueeze(0), synth_feat.unsqueeze(0), dim=-1).squeeze(0)
    cos_dist = (1.0 - cos_sim).clamp(0.0, 2.0)
    dist_up = F.interpolate(cos_dist.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().numpy()
    p99 = float(np.percentile(dist_up, 99))
    return np.clip(dist_up / p99, 0.0, 1.0).astype(np.float32) if p99 > 0 else np.zeros_like(dist_up, dtype=np.float32)

def geo_score(synth_np, warped_np, mask_np):                           # cell 2 verbatim (via recompute_geo.py)
    H, W = mask_np.shape; score_map = np.ones((H, W), dtype=np.float32); visible = mask_np >= 0.5
    if visible.any():
        diff = np.abs(synth_np - warped_np).mean(axis=2); p99 = np.percentile(diff[visible], 99)
        diff_norm = np.clip(diff / p99, 0.0, 1.0) if p99 > 0 else np.zeros_like(diff); score_map[visible] = diff_norm[visible]
    return score_map
to_np = lambda pil: np.array(pil, dtype=np.float32) / 255.

def main():
    ap = argparse.ArgumentParser()
    for a_ in ("--synth-dir", "--source-dir", "--warp-dir", "--mask-dir", "--out"): ap.add_argument(a_, required=True)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    device = a.device
    dino = torch.hub.load(f"facebookresearch/dinov2:{DINOV2_COMMIT}", "dinov2_vits14", skip_validation=True).to(device).eval()
    dino_transform = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    files = sorted(f for f in os.listdir(a.synth_dir) if f.lower().endswith((".jpg", ".png")))
    rows = []; t0 = time.time()
    for i, f in enumerate(files):
        stem, ele, rad = parse(f); src_path = os.path.join(a.source_dir, stem + ".jpg")
        wp = os.path.join(a.warp_dir, os.path.splitext(f)[0] + ".jpg"); mp = os.path.join(a.mask_dir, os.path.splitext(f)[0] + "_mask.png")
        for p in (src_path, wp, mp):
            if not os.path.exists(p): sys.exit(f"missing {p}")
        src_pil = Image.open(src_path).convert("RGB"); synth_pil = Image.open(os.path.join(a.synth_dir, f)).convert("RGB")
        orig_w, orig_h = src_pil.size
        sm = compute_semantic_score(src_pil, synth_pil, dino, dino_transform, device, orig_h, orig_w); sem = float(sm.mean())
        warped = Image.open(wp).convert("RGB"); mask = Image.open(mp).convert("L")
        if warped.size != synth_pil.size: warped = warped.resize(synth_pil.size, Image.BILINEAR)
        if mask.size != synth_pil.size: mask = mask.resize(synth_pil.size, Image.NEAREST)
        mask_np = to_np(mask); g = float(geo_score(to_np(synth_pil), to_np(warped), mask_np).mean()); hole = float((mask_np < 0.5).mean())
        rows.append(dict(image_stem=stem, synth_file=f, source_file=stem + ".jpg", ele=f"{ele:g}", r=f"{rad:.2f}", holefrac=f"{hole:.4f}",
                         mean_sem_score=f"{sem:.4f}", geo_fixed=f"{g:.4f}", mean_combined_fixed=f"{(float(f'{g:.4f}') + float(f'{sem:.4f}'))/2:.4f}"))
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
