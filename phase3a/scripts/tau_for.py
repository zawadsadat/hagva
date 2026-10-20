#!/home/newview/miniconda3/envs/vprtutorial/bin/python3
"""tau_for.py <scores_csv> <column> <n> [<n> ...]
For each n print: n tau expected_n_pass tie
tau = midpoint between n-th and (n+1)-th lowest value of the column (strict '<' in the gate).
n=400 (all views): tau = max + 1.0.
"""
import csv, sys
import numpy as np
path, col = sys.argv[1], sys.argv[2]
v = np.sort(np.array([float(r[col]) for r in csv.DictReader(open(path, newline=""))]))
for n in map(int, sys.argv[3:]):
    if n >= len(v):
        tau = float(v[-1]) + 1.0; tie = False
    else:
        tau = (float(v[n-1]) + float(v[n])) / 2.0; tie = bool(v[n-1] == v[n])
    print(n, repr(tau), int((v < tau).sum()), int(tie))
