import csv

DATASETS = {
    'gp':      '/mnt/sda/Zawad/genwarp/hagva_module_a/gardenspoint_400/validation/combined_summary.csv',
    'sfu':     '/mnt/sda/Zawad/genwarp/hagva_module_a/sfu_400/validation/combined_summary.csv',
    'stlucia': '/mnt/sda/Zawad/genwarp/hagva_module_a/stlucia_400/validation/combined_summary.csv',
}
K_PERCENTS = [1, 2.5, 5, 10, 25]

for name, path in DATASETS.items():
    scores = []
    with open(path) as f:
        for row in csv.DictReader(f):
            scores.append(float(row['mean_combined_score']))
    scores.sort()
    n = len(scores)
    print(f"\n{name}  (n={n})")
    for k in K_PERCENTS:
        cnt = max(1, round(n * k / 100))
        # midpoint between cnt-th and (cnt+1)-th lowest -> admits exactly cnt
        tau = (scores[cnt-1] + scores[cnt]) / 2 if cnt < n else scores[-1] + 1e-6
        print(f"  k={k:>4}%  views={cnt:>3}  tau={tau:.6f}")
