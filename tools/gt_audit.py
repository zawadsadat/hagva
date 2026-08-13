"""
Audit GT completeness for a staged dataset.
Usage: gt_audit.py <framework_dir> <DatasetClass> <staging_images_dir>
Reports: query count, columns with GT, columns all-zero, first zero index.
"""
import sys, os
sys.path.insert(0, sys.argv[1])
import datasets.load_dataset as L

cls_name, stage = sys.argv[2], sys.argv[3]
cls = getattr(L, cls_name)
db, q, GThard, GTsoft = cls(destination=stage).load()

n  = GThard.shape[1]
nz = int((GThard.sum(axis=0) > 0).sum())
first = next((i for i in range(n) if GThard[:, i].sum() == 0), None)
print(f"refs={GThard.shape[0]}  queries={n}  with_GT={nz}  all_zero={n-nz}  first_zero={first}")
import numpy as _np, os as _os
_gtp = _os.path.join(stage, "GT.npz")
if _os.path.exists(_gtp):
    _H = _np.load(_gtp)["GThard"].astype(bool)
    _base_zero = int((_H.sum(axis=0) == 0).sum())
    print(f"original_all_zero={_base_zero}")
    print("STATUS:", "OK" if (n - nz) <= _base_zero else "*** INCOMPLETE GT ***")
else:
    print("STATUS:", "OK" if nz == n else "*** INCOMPLETE GT ***")
