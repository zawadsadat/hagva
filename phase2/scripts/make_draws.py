#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""
Phase 2 (c): five seeded draws of 50 views from the 400-view pool.  CPU only, no images touched.

  make_draws.py [--ds gp|sfu|stl] [--seeds 0 1 2 3 4] [--n 50] [--out /mnt/sda/HAGVA_R2/phase2/draws]

Pool = the 400 synthesized views listed in module_a/scores/<ds>_400_signal1_summary.csv
(sorted by synth_file so the stem order is deterministic). Draw k:
    numpy.random.default_rng(seed).choice(stems, n, replace=False)
One CSV per seed: <out>/<ds>_seed<k>.csv with
    seed,k,stem,synth_file,source_file,source_image,ref_idx,ele_deg,radius
    stem         = '000_Image127' (pool stem, = source_file without extension)
    source_image = 'Image127.jpg' (the real query the view was synthesized from)
    ref_idx      = GardensPoint reference index = int(NNN) of ImageNNN (identity GT); '' for other ds
Also writes <out>/draws_summary.md with the pool check requested in the brief: number of distinct
source images in the 400 pool (the brief expects 200), the multiplicity histogram, and for each
draw how many distinct sources it covers and how many sources appear more than once in it, plus
pairwise overlap between draws.  The 200-distinct expectation is REPORTED, not asserted, because
the GardensPoint pool actually has 172 distinct sources (see inventory / summary).
"""
import argparse, csv, os, re, collections, itertools
import numpy as np

SCORES = "/mnt/sda/hagva-repo/module_a/scores"
DS = {"gp": "gardenspoint", "sfu": "sfu", "stl": "stlucia"}

def ref_idx_gp(source_image):
    m = re.search(r"Image(\d+)", source_image)
    return int(m.group(1)) if m else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="gp", choices=DS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="/mnt/sda/HAGVA_R2/phase2/draws")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(f"{SCORES}/{DS[a.ds]}_400_signal1_summary.csv", newline="")))
    rows.sort(key=lambda r: r["synth_file"])
    assert len(rows) == 400, len(rows)
    by_stem = {}
    for r in rows:
        stem = os.path.splitext(r["source_file"])[0]                 # '000_Image127'
        src_img = re.sub(r"^\d+_", "", r["source_file"], count=1)     # 'Image127.jpg'
        assert stem not in by_stem, f"duplicate stem {stem}"
        by_stem[stem] = dict(synth_file=r["synth_file"], source_file=r["source_file"], source_image=src_img,
                             ref_idx=ref_idx_gp(src_img) if a.ds == "gp" else "",
                             ele_deg=r["ele_deg"], radius=r["radius"])
    stems = sorted(by_stem)                                           # deterministic order
    pool_sources = collections.Counter(v["source_image"] for v in by_stem.values())
    mult_hist = sorted(collections.Counter(pool_sources.values()).items())

    L = [f"# Draws for {DS[a.ds]} ({a.n} of 400, seeds {a.seeds})\n",
         f"Pool: `{SCORES}/{DS[a.ds]}_400_signal1_summary.csv`, 400 views, **{len(pool_sources)} distinct source images** "
         f"(brief expected 200 → {'OK' if len(pool_sources) == 200 else 'MISMATCH: the pool was drawn WITH replacement from the 200 queries'}).",
         f"Source multiplicity histogram (views per source → #sources): {mult_hist}\n",
         "| seed | views | distinct sources | sources with ≥2 views in draw | max views of one source | distinct ref_idx |", "|---|---|---|---|---|---|"]
    draws = {}
    for s in a.seeds:
        rng = np.random.default_rng(s)
        pick = [str(x) for x in rng.choice(np.array(stems), size=a.n, replace=False)]
        draws[s] = pick
        path = f"{a.out}/{a.ds}_seed{s}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["seed", "k", "stem", "synth_file", "source_file", "source_image", "ref_idx", "ele_deg", "radius"])
            for k, st in enumerate(pick):
                v = by_stem[st]; w.writerow([s, k, st, v["synth_file"], v["source_file"], v["source_image"], v["ref_idx"], v["ele_deg"], v["radius"]])
        srcs = collections.Counter(by_stem[st]["source_image"] for st in pick)
        L.append(f"| {s} | {len(pick)} | {len(srcs)} | {sum(1 for c in srcs.values() if c > 1)} | {max(srcs.values())} | {len(set(by_stem[st]['ref_idx'] for st in pick))} |")
        print(f"seed {s}: wrote {path}; distinct sources {len(srcs)}, repeated sources {sum(1 for c in srcs.values() if c > 1)}")
    L.append("\nPairwise view overlap between draws (shared stems):\n")
    L.append("| seeds | shared stems | shared sources |"); L.append("|---|---|---|")
    for s1, s2 in itertools.combinations(a.seeds, 2):
        sh = set(draws[s1]) & set(draws[s2])
        shs = set(by_stem[x]["source_image"] for x in draws[s1]) & set(by_stem[x]["source_image"] for x in draws[s2])
        L.append(f"| {s1},{s2} | {len(sh)} | {len(shs)} |")
    L.append("\nNotes: a source that appears twice in a draw is injected twice (two different synthesized views, two query slots, "
             "both with GT = the same reference index). For C2 (real-image control) the same source image is then injected twice; "
             "for C2b the same day_left image is injected twice.")
    open(f"{a.out}/draws_summary.md", "w").write("\n".join(L) + "\n")
    print(f"pool distinct sources: {len(pool_sources)}  (expected 200: {'OK' if len(pool_sources)==200 else 'MISMATCH'})  -> {a.out}/draws_summary.md")

if __name__ == "__main__":
    main()
