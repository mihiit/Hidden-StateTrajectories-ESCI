"""Simulations behind the revised paper: null AUROC spread, sampling spread, paired-AUROC power (DeLong)."""
import numpy as np, json, sys
from scipy.stats import norm
rng = np.random.default_rng(2026)

def wilson(k, n, z=1.959964):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h, c+h

def auc(pos, neg):
    # Mann-Whitney AUROC with ties=0.5
    return ((pos[:,None] > neg[None,:]).sum() + 0.5*(pos[:,None] == neg[None,:]).sum())/(len(pos)*len(neg))

def delong_paired(p1, n1, p2, n2):
    """Paired DeLong test for two correlated AUROCs on same samples. Returns (a1,a2,z,p)."""
    m, n = len(p1), len(n1)
    def comps(p, q):
        psi = (p[:,None] > q[None,:]) + 0.5*(p[:,None] == q[None,:])
        return psi.mean(1), psi.mean(0), psi.mean()
    v10_1, v01_1, a1 = comps(p1, n1); v10_2, v01_2, a2 = comps(p2, n2)
    s10 = np.cov(np.vstack([v10_1, v10_2])); s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10/m + s01/n
    var = S[0,0] + S[1,1] - 2*S[0,1]
    z = (a1-a2)/np.sqrt(var) if var > 0 else 0.0
    return a1, a2, z, 2*(1-norm.cdf(abs(z)))

def d_for_auc(a): return np.sqrt(2)*norm.ppf(a)

def draw(npos, nneg, a1, a2, rho):
    d1, d2 = d_for_auc(a1), d_for_auc(a2)
    C = np.array([[1, rho], [rho, 1]])
    neg = rng.multivariate_normal([0,0], C, nneg)
    pos = rng.multivariate_normal([d1,d2], C, npos)
    return pos, neg

out = {}
# 0. Verify numbers quoted in the paper
out['wilson_3_24'] = wilson(3,24); out['wilson_4_179'] = wilson(4,179); out['wilson_4_40']=wilson(4,40)
out['wilson_3_24_pct'] = [round(100*x,1) for x in out['wilson_3_24']]
praw = np.array([0.048,0.160,0.126]); order = np.argsort(praw); holm = np.empty(3); run = 0
for rank, i in enumerate(order):
    run = max(run, (3-rank)*praw[i]); holm[i] = min(1, run)
out['holm'] = holm.tolist()
out['se_from_ci'] = 0.489/(2*1.96)
out['sign_test_3of3'] = 0.5**3

# 1. Null AUROC distribution at the current test fold (8 pos / 52 neg) and sampling spread of a true-0.776 detector
NPOS, NNEG = 8, 52
nulls = np.array([auc(rng.normal(size=NPOS), rng.normal(size=NNEG)) for _ in range(20000)])
alt = []
d = d_for_auc(0.776)
for _ in range(20000):
    alt.append(auc(rng.normal(d,1,NPOS), rng.normal(0,1,NNEG)))
alt = np.array(alt)
out['null_cur'] = dict(mean=nulls.mean(), sd=nulls.std(), q025=np.quantile(nulls,.025), q975=np.quantile(nulls,.975))
out['alt_cur'] = dict(mean=alt.mean(), sd=alt.std(), q025=np.quantile(alt,.025), q975=np.quantile(alt,.975))
out['p_null_ge_0776'] = float((nulls >= 0.776).mean())
out['p_null_le_0264'] = float((nulls <= 0.264).mean()); out['p_null_ge_0651'] = float((nulls >= 0.651).mean())
# planned: pooled out-of-fold 25 pos / 175 neg, and expanded 63 pos / 437 neg
def spread(npos, nneg, a=0.776, reps=4000):
    dd = d_for_auc(a)
    v = np.array([auc(rng.normal(dd,1,npos), rng.normal(0,1,nneg)) for _ in range(reps)])
    n0 = np.array([auc(rng.normal(size=npos), rng.normal(size=nneg)) for _ in range(reps)])
    return v, n0
S = {}
for name,(npos,nneg) in {'cur_fold':(8,52),'pooled_oof_200':(25,175),'n500':(63,437)}.items():
    v,n0 = spread(npos,nneg); S[name]=dict(npos=npos,nneg=nneg,alt_q=[np.quantile(v,.025),np.quantile(v,.975)],alt_sd=v.std(),null_q=[np.quantile(n0,.025),np.quantile(n0,.975)],null_sd=n0.std())
    np.save(f'/home/claude/work/figs/alt_{name}.npy', v); np.save(f'/home/claude/work/figs/null_{name}.npy', n0)
out['spread'] = S

# 2. Precision of AUROC half-width vs number of positives (positive rate fixed 12.5%)
grid_pos = [8,15,25,40,63,100,150]
hw = []
for npos in grid_pos:
    nneg = int(round(npos*7)); v,_ = spread(npos,nneg,reps=1500)
    hw.append([npos, (np.quantile(v,.975)-np.quantile(v,.025))/2])
out['halfwidth'] = hw

# 3. Power of paired DeLong test, RTM AUROC=0.78 vs comparator AUROC = 0.78-delta, rho within-class score correlation
RHO = 0.3
Ns = [200,300,400,500,700,900,1200,1600]
power = {}
for delta in [0.10,0.15,0.20]:
    for alpha in [0.05, 0.05/3]:
        key = f'd{delta}_a{alpha:.4f}'; row=[]
        for N in Ns:
            npos = int(round(N*0.125)); nneg = N-npos; hits = 0; reps = 600
            for _ in range(reps):
                pos, neg = draw(npos,nneg,0.78,0.78-delta,RHO)
                a1,a2,z,p = delong_paired(pos[:,0],neg[:,0],pos[:,1],neg[:,1])
                hits += (p<alpha) and (a1>a2)
            row.append(hits/reps)
        power[key]=row; print(key,row,flush=True)
out['power']=dict(Ns=Ns,rho=RHO,curves=power)
# 4. Type-I check of the paired test at N=200 and N=500 (delta=0)
t1 = {}
for N in [200,500]:
    npos=int(round(N*.125)); hits=0; reps=1500
    for _ in range(reps):
        pos,neg=draw(npos,N-npos,0.78,0.78,RHO); a1,a2,z,p=delong_paired(pos[:,0],neg[:,0],pos[:,1],neg[:,1]); hits+=(p<0.05)
    t1[N]=hits/reps
out['type1_delong']=t1
json.dump(out, open('/home/claude/work/figs/sim_results.json','w'), indent=1, default=float)
print(json.dumps({k:v for k,v in out.items() if k!='power'}, indent=1, default=float))
