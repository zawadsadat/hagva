import csv
import numpy as np

DATASETS = {
    'gp':      '/mnt/sda/Zawad/genwarp/hagva_module_a/gardenspoint_400/validation/combined_summary.csv',
    'sfu':     '/mnt/sda/Zawad/genwarp/hagva_module_a/sfu_400/validation/combined_summary.csv',
    'stlucia': '/mnt/sda/Zawad/genwarp/hagva_module_a/stlucia_400/validation/combined_summary.csv',
}
SIGMAS = [1.0, 1.25, 1.5, 1.75, 2.0]

for name, path in DATASETS.items():
    s = []
    with open(path) as f:
        for row in csv.DictReader(f):
            s.append(float(row['mean_combined_score']))
    s = np.array(s)
    mu, sd = s.mean(), s.std()
    print(f"\n{name}:  mean={mu:.4f}  std={sd:.4f}  min={s.min():.4f}  max={s.max():.4f}")
    for m in SIGMAS:
        tau = mu - m * sd
        n = int((s < tau).sum())
        print(f"   mean-{m:.2f}s -> tau={tau:.4f}   accepted={n:>3}  ({100*n/len(s):.1f}%)")
