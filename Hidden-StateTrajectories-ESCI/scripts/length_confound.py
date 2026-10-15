"""Can response length alone make a trajectory detector look informative?
Simulation: per-token features are pure noise (no label information). Labels depend only on response length.
Features are summarised exactly as RTM does (mean / max / last over generated tokens, per layer).
We report out-of-fold AUROC of (i) length only, (ii) the summary features, (iii) the summary features after
regressing log-length out of every feature (the control proposed in the protocol)."""
import numpy as np, json
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(5)
N, PREV, LAYERS, REPS = 900, 0.125, 12, 60
def d_for_auc(a): return np.sqrt(2) * norm.ppf(a)

def make(a_len):
    y = (rng.random(N) < PREV).astype(int)
    # reliable answers are shorter; separation chosen so length alone has AUROC a_len
    ll = rng.normal(4.0, 0.6, N) - y * d_for_auc(a_len) * 0.6          # log tokens
    T = np.clip(np.exp(ll).astype(int), 8, 256)
    F = []
    for t in T:
        x = rng.normal(size=(LAYERS, t))                              # token-axis feature, no label info
        F.append(np.concatenate([x.mean(1), x.max(1), x[:, -1]]))
    return y, np.log(T), np.array(F)

def oof(X, y):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return roc_auc_score(y, s)

def resid(X, z, tr):                                                  # remove log-length, fit on training rows only
    lr = LinearRegression().fit(z[tr, None], X[tr]); return X - lr.predict(z[:, None])

def oof_resid(X, z, y):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        R = resid(X, z, tr)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000)).fit(R[tr], y[tr])
        s[te] = m.decision_function(R[te])
    return roc_auc_score(y, s)

levels = [0.5, 0.6, 0.7, 0.8]; out = {"length_auc": levels, "len_only": [], "features": [], "features_resid": []}
for a in levels:
    r = [];
    for _ in range(REPS):
        y, z, F = make(a); r.append([oof(z[:, None], y), oof(F, y), oof_resid(F, z, y)])
    r = np.array(r); out["len_only"].append(r[:, 0].tolist()); out["features"].append(r[:, 1].tolist()); out["features_resid"].append(r[:, 2].tolist())
    print(f"length AUROC {a}: length-only {r[:,0].mean():.3f}  features(no content) {r[:,1].mean():.3f} [{np.quantile(r[:,1],.025):.3f},{np.quantile(r[:,1],.975):.3f}]  after length removed {r[:,2].mean():.3f}")
json.dump(out, open("figs/length_confound.json", "w"))
