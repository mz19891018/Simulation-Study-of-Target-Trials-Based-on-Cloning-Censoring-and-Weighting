import numpy as np, pandas as pd, json
from ccw import *
from multiprocessing import Pool
raw=pd.read_csv("cohort2.csv"); P=prep(raw)
def f(seed):
    rng=np.random.default_rng(abs(seed))
    d0=P if seed<0 else P.iloc[rng.integers(0,len(P),len(P))].reset_index(drop=True)
    d=impute(d0,abs(seed)+7); dz=Design(d); g=(d.mmae_day.values<=7)
    out={}
    # naive: treatment received, time zero day 0, unweighted
    one=np.ones((len(d),T+1)); big=np.full(len(d),10**6)
    a=cif_weighted(d[g].reset_index(drop=True),one[g],big[g],"reop")[0][180]; b=cif_weighted(d[~g].reset_index(drop=True),one[~g],big[~g],"reop")[0][180]
    out["naive"]=[a,b]
    # landmark day 8 + IPTW (original design)
    lm=at_risk_day(d,8); dl=d[lm].reset_index(drop=True); gl=g[lm]
    X=dz.X(dl); m=fit(X,gl); ps=m.predict_proba(X)[:,1]; pt=gl.mean(); w=np.where(gl,pt/ps,(1-pt)/(1-ps))
    W=np.repeat(w[:,None],T+1,1)
    a=cif_weighted(dl[gl].reset_index(drop=True),W[gl],big[:gl.sum()],"reop")[0][180]; b=cif_weighted(dl[~gl].reset_index(drop=True),W[~gl],big[:(~gl).sum()],"reop")[0][180]
    out["landmark"]=[a,b]
    return out
if __name__=="__main__":
    pt=f(-1)
    with Pool(2) as p: bs=p.map(f,range(9000,9200))
    R={}
    for k in ("naive","landmark"):
        rd=[b[k][0]-b[k][1] for b in bs]; rr=[b[k][0]/b[k][1] for b in bs]
        R[k]={"a":pt[k][0],"b":pt[k][1],"rd":pt[k][0]-pt[k][1],"rd_ci":list(np.percentile(rd,[2.5,97.5])),"rr":pt[k][0]/pt[k][1],"rr_ci":list(np.percentile(rr,[2.5,97.5]))}
    json.dump(R,open("naive.json","w"),default=float); print(R)
