#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""Wide-pool draws: same seeds and the same stem sampling as phase2/scripts/make_draws.py (default_rng(seed).choice over the sorted
400 pool stems, replace=False), but synth_file / ele / radius taken from phase3b/scores/<ds>_wide_scores.csv. Writes phase3b/draws/<ds>_seed<k>.csv
and asserts the stem sets equal the Phase 2 draws (so C2 is identical across pools)."""
import csv, os, re, numpy as np
R="/mnt/sda/HAGVA_R2"; DS={"stl":"stlucia"}   # 3b-STL
for ds,name in DS.items():
    W={r["image_stem"]:r for r in csv.DictReader(open(f"{R}/phase3b/scores/{name}_wide_scores.csv"))}; stems=sorted(W); assert len(stems)==400
    for s in range(5):
        pick=[str(x) for x in np.random.default_rng(s).choice(np.array(stems), size=50, replace=False)]
        ref=[r["stem"] for r in csv.DictReader(open(f"{R}/phase2/draws/{ds}_seed{s}.csv"))]; assert pick==ref, (ds,s)
        with open(f"{R}/phase3b/draws/{ds}_seed{s}.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["seed","k","stem","synth_file","source_file","source_image","ref_idx","ele_deg","radius"])
            for k,st in enumerate(pick):
                r=W[st]; src=re.sub(r"^\d+_","",r["source_file"],count=1); m=re.search(r"Image(\d+)",src)
                w.writerow([s,k,st,r["synth_file"],r["source_file"],src,int(m.group(1)) if (ds=="gp" and m) else "",r["ele"],r["r"]])
        print(f"{ds} seed{s}: 50 stems identical to phase2 draw; wide synth files e.g. {W[pick[0]]['synth_file']}")
