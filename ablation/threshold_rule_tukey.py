import csv
import numpy as np

DATASETS = {
    'gp':      '/mnt/sda/Zawad/genwarp/hagva_module_a/gardenspoint_400/validation/combined_summary.csv',
    'sfu':     '/mnt/sda/Zawad/genwarp/hagva_module_a/sfu_400/validation/combined_summary.csv',
    'stlucia': '/mnt/sda/Zawad/genwarp/hagva_module_a/stlucia_400/validation/combined_summary.csv',
}

for name, path in DATASETS.items():
    s = []
    with open(path) as f:
        for row in csv.DictReader(f):
            s.append(float(row['mean_combined_score']))
    s = np.array(s)
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    iqr = q3 - q1
    print(f"\n{name}:  Q1={q1:.4f}  median={q2:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}  min={s.min():.4f}")
    for label, tau in [
        ("Q1 - 1.5*IQR", q1 - 1.5*iqr),
        ("Q1 - 1.0*IQR", q1 - 1.0*iqr),
        ("Q1 - 0.5*IQR", q1 - 0.5*iqr),
        ("Q1          ", q1),
        ("10th pctile ", np.percentile(s, 10)),
        ("5th pctile  ", np.percentile(s, 5)),
    ]:
        n = int((s < tau).sum())
        print(f"   {label} -> tau={tau:.4f}   accepted={n:>3}  ({100*n/len(s):.1f}%)")
