import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
import ccw3
from ccw3 import *
raw=pd.read_csv("cohort3.csv"); P=prep(raw)
OUTS=("reop","any","death","te","pneu")
KEYBAL=["age","thick","resid_ml","pneumo_ml","sep","bilateral","hyperdense","anticoag","high_thromb","egfr_lt60","pre_mrs2","mw34","yr","centerA","reversal","two_burr","subgaleal","dm","af","male","htn","chf","alcohol","plt","inr","atorva","dapt","apac","vka","prior_stroke","mls","nk_sep","nk_tra","nk_lam"]
def balance(d,wa,wb):
    r8=at_risk_day(d,8); g=d.mmae_day.values<=7; out={}
    for col in KEYBAL:
        if col not in d or d[col].nunique()<2: continue
        x=d[col].values.astype(float); sd=np.sqrt((x[g].var()+x[~g].var())/2)
        if sd>0 and (wa*r8).sum()>0 and (wb*r8).sum()>0: out[col]=(np.average(x,weights=wa*r8)-np.average(x,weights=wb*r8))/sd
    return out
def est(d,o):
    G=o.get("G",7); dz=Design3(d,reduced=o.get("reduced",False),no_ct=o.get("no_ct",False))
    WA,cA,WB,cB,S=weights(d,dz,G,ct_day1=o.get("ct_day1",False),trunc=o.get("trunc"),eb_A=o.get("eb",False),calib=o.get("calib",True))
    R={}
    for k,(W,c) in {"A":(WA,cA),"B":(WB,cB)}.items():
        R[k]={x:cif_weighted(d,W,c,x)[0] for x in OUTS}; R[k]["curve"]=cif_weighted(d,W,c,"reop")[1]
        w=day8_weight(d,W,c,G); R[k]["w"]=w
        for y,ob in (("rad","rad_obs"),("mrs02","mrs_obs")):
            ow=obs_weight(d,dz,ob); val=(d.rad.values if y=="rad" else (d.mrs180.values<=2)).astype(float)
            ww=w*ow; R[k][y]=np.average(np.nan_to_num(val),weights=ww)
        R[k]["ess"]=w.sum()**2/(w**2).sum(); R[k]["wmax"]=w.max()
    b=balance(d,R["A"]["w"],R["B"]["w"]); R["maxsmd"]=max(abs(v) for v in b.values()); R["bal"]=b
    return R
def contrast(R):
    out={}
    for x in OUTS:
        for g in (30,90,180): out[f"{x}_rd{g}"]=R["A"][x][g]-R["B"][x][g]
        out[f"{x}_rr"]=R["A"][x][180]/R["B"][x][180]; out[f"{x}_a"]=R["A"][x][180]; out[f"{x}_b"]=R["B"][x][180]
        out[f"{x}_rmtl"]=R["A"][x]["rmtl"]-R["B"][x]["rmtl"]
    for y in ("rad","mrs02"): out[y+"_a"]=R["A"][y]; out[y+"_b"]=R["B"][y]; out[y+"_rd"]=R["A"][y]-R["B"][y]
    out["essA"]=R["A"]["ess"]; out["essB"]=R["B"]["ess"]; out["wmaxA"]=R["A"]["wmax"]; out["wmaxB"]=R["B"]["wmax"]; out["maxsmd"]=R["maxsmd"]
    return out
def data_for(o,seed):
    if seed is None: d0=P
    else:
        idx=np.random.default_rng(seed).integers(0,len(P),len(P)); d0=P.iloc[idx].reset_index(drop=True)
    if o.get("cc"): d=d0.dropna(subset=BASE).reset_index(drop=True)
    else: d=impute(d0,(seed or 0)+o.get("mseed",100))
    f=o.get("filter")
    if f=="no_particle": d=d[~((d.mmae_day<=7)&(d.agent=="颗粒+弹簧圈"))]
    if f=="centerA": d=d[d.center=="A"]
    if f=="centerB": d=d[d.center=="B"]
    if f=="excl_preCT": d=d[~((d.mmae_day==1)&(d.mmae_h<d.ct1_h))]
    if "subgroup" in o: c,l=o["subgroup"]; d=d[d[c]==l]
    return d.reset_index(drop=True)
def joint_est(d):
    dz=Design3(d); Hm=hazard_model(d,dz); Sm=np.cumprod(1-Hm,axis=1); Hr=res_hazard(d,dz); Sr=np.cumprod(1-Hr,axis=1)
    out={}
    for k,(sm,sr) in {"MR":(1,1),"Mr":(1,0),"mR":(0,1),"mr":(0,0)}.items():
        W1,c1=seq_w(d.mmae_day.values,Sm,sm); W2,c2=seq_w(d.res_day.values,Sr,sr)
        W=W1*W2; c=np.minimum(c1,c2); w=day8_weight(d,W,c)
        out[k]={x:cif_weighted(d,W,c,x)[0][180] for x in ("reop","te","death")}
        out[k]["ess"]=w.sum()**2/(w**2).sum(); out[k]["n_follow"]=int((w>0).sum()); out[k]["wmax"]=float(w.max())
    return out
def job(a):
    o,seed=a; d=data_for(o,seed)
    if o.get("joint"): return joint_est(d)
    R=est(d,o); c=contrast(R)
    if o.get("curves"): c["curveA"]=R["A"]["curve"].tolist(); c["curveB"]=R["B"]["curve"].tolist()
    return c
def point(o,M=10):
    outs=[]
    for m in range(M):
        oo={**o,"mseed":100+m}; d=data_for(oo,None) if not o.get("cc") else data_for(o,None)
        if o.get("joint"): outs.append(joint_est(d))
        else:
            R=est(d,o); c=contrast(R); c["curveA"]=R["A"]["curve"]; c["curveB"]=R["B"]["curve"]; c["bal"]=R["bal"]; outs.append(c)
        if o.get("cc"): break
    if o.get("joint"): return {k:{x:float(np.mean([z[k][x] for z in outs])) for x in outs[0][k]} for k in outs[0]}
    res={k:np.mean([z[k] for z in outs],axis=0) for k in outs[0] if k!="bal"}
    res["bal"]={k:float(np.mean([z["bal"][k] for z in outs])) for k in outs[0]["bal"]}
    return res
def boot(o,B,seed0,procs=2):
    with Pool(procs) as p: return p.map(job,[(o,seed0+i) for i in range(B)])
def ci(v): v=np.array([x for x in v if x==x]); return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
