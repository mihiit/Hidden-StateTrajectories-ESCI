#!/usr/bin/env python3
"""Step 1 of the v2 analyses: read the hidden-state files ONCE and cache the feature building blocks.

    python scripts/featurize.py --feats-dir acts/ --labels annotation/labels.csv --out cache/features.npz

The cache is a few MB. All later scripts (stats_robustness.py, length_adjusted_eval.py) read it instead of the 15-20 GB
of activations.  Integrity check: the first --check items are re-assembled and compared with rtm_protocol.families.
"""
import argparse, os, sys
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_common as C  # noqa: E402
import rtm_protocol as P  # noqa: E402


def one(path):
    h = np.load(path)["hidden"]
    return C.blocks(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats-dir", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-jobs", type=int, default=1); ap.add_argument("--check", type=int, default=5)
    a = ap.parse_args()
    ids, y = C.load_labels(a.labels)
    paths = [os.path.join(a.feats_dir, i + ".npz") for i in ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing: sys.exit(f"{len(missing)} activation files missing, first: {missing[0]}")
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(p) for p in paths)
    B = {k: np.vstack([r[k] for r in res]) for k in res[0]}
    # integrity: the cached blocks must rebuild exactly the families the confirmatory run used
    for j in range(min(a.check, len(ids))):
        fam = P.families(np.load(paths[j])["hidden"])
        for name in C.PRIMARY:
            row = C.assemble({k: v[j:j + 1] for k, v in B.items()}, name)[0]
            assert np.allclose(row, fam[name], rtol=1e-9, atol=1e-9), f"family {name} differs for item {ids[j]}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, ids=np.array(ids), y=y, **B)
    nt = B["ntok"].ravel()
    print(f"cached {len(ids)} items; reliable {int(y.sum())}; tokens min/median/max {int(nt.min())}/{int(np.median(nt))}/{int(nt.max())}; integrity check OK")


if __name__ == "__main__":
    main()
