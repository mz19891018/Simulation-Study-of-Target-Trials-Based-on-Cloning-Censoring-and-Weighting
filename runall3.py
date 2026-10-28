import json, numpy as np, time
from scipy.stats import norm
from full3 import *
def S(x): return x.tolist() if isinstance(x,np.ndarray) else (float(x) if not isinstance(x,dict) else x)
if __name__=="__main__":
    R={}; t0=time.time()
    pt=point({},M=10); bs=boot({"curves":True},500,1000)
    R["main"]={"pt":{k:S(v) for k,v in pt.items()},"ci":{k:ci([b[k] for b in bs]) for k in bs[0] if not k.startswith("curve")}}
    for side in "AB":
        c=np.array([b["curve"+side] for b in bs]); R["main"]["band"+side]=[np.percentile(c,2.5,0).tolist(),np.percentile(c,97.5,0).tolist()]
    R["main"]["p"]={}
    for k in ["reop_rd180","any_rd180","death_rd180","te_rd180","pneu_rd180","rad_rd","mrs02_rd","reop_rd30","reop_rd90"]:
        se=np.std([b[k] for b in bs]); R["main"]["p"][k]=float(2*(1-norm.cdf(abs(pt[k])/se)))
    print("main",time.time()-t0,flush=True); json.dump(R,open("R3.json","w"))
    SENS={"nocal":{"calib":False},"eb":{"eb":True},"trunc99":{"trunc":99},"trunc95":{"trunc":95},"ct_day1":{"ct_day1":True,"filter":"excl_preCT"},
          "excl_preCT":{"filter":"excl_preCT"},"no_ct":{"no_ct":True},"grace5":{"G":5},"grace14":{"G":14},"reduced":{"reduced":True},
          "cc":{"cc":True},"no_particle":{"filter":"no_particle"},"centerA":{"filter":"centerA","reduced":True},"centerB":{"filter":"centerB","reduced":True}}
    R["sens"]={}
    for k,o in SENS.items():
        pt=point(o,M=5); bs=boot(o,200,2000+len(k)*31)
        R["sens"][k]={"pt":{x:S(pt[x]) for x in ("reop_rd180","reop_rr","reop_a","reop_b","essA","essB","wmaxA","wmaxB","maxsmd")},
                      "ci":{x:ci([b[x] for b in bs]) for x in ("reop_rd180","reop_rr")}}
        print(k,R["sens"][k]["pt"]["reop_rd180"],time.time()-t0,flush=True); json.dump(R,open("R3.json","w"))
    R["sub"]={}
    for col in ("anticoag","high_thromb","sep","age75"):
        R["sub"][col]={}; bl={}
        for lev in (0,1):
            o={"subgroup":(col,lev),"reduced":True}; pt=point(o,M=5); bs=boot(o,200,3000+lev*500+len(col)*7); bl[lev]=bs
            R["sub"][col][str(lev)]={"pt":{x:S(pt[x]) for x in ("reop_rd180","reop_rr","reop_a","reop_b","maxsmd","essA","essB")},
               "ci":{x:ci([b[x] for b in bs]) for x in ("reop_rd180","reop_rr")},"n":int((P[col]==lev).sum()),"nA":int(((P[col]==lev)&(P.mmae_day<=7)).sum())}
        diff=[a["reop_rd180"]-b["reop_rd180"] for a,b in zip(bl[1],bl[0])]
        dpt=R["sub"][col]["1"]["pt"]["reop_rd180"]-R["sub"][col]["0"]["pt"]["reop_rd180"]
        R["sub"][col]["inter_rd"]=[dpt]+ci(diff)+[float(2*(1-norm.cdf(abs(dpt)/np.std(diff))))]
        print(col,time.time()-t0,flush=True); json.dump(R,open("R3.json","w"))
    o={"joint":True}; pt=point(o,M=5); bs=boot(o,300,5000)
    R["joint"]={"pt":pt,"ci":{k:{x:ci([b[k][x] for b in bs]) for x in ("reop","te","death")} for k in pt}}
    for a,b,nm in [("MR","mR","early"),("Mr","mr","late")]:
        for x in ("reop","te"): R["joint"][f"{nm}_{x}_rd"]=[pt[a][x]-pt[b][x]]+ci([bb[a][x]-bb[b][x] for bb in bs])
    v=[(bb["MR"]["reop"]-bb["mR"]["reop"])-(bb["Mr"]["reop"]-bb["mr"]["reop"]) for bb in bs]
    dd=(pt["MR"]["reop"]-pt["mR"]["reop"])-(pt["Mr"]["reop"]-pt["mr"]["reop"]); R["joint"]["inter"]=[dd]+ci(v)+[float(2*(1-norm.cdf(abs(dd)/np.std(v))))]
    json.dump(R,open("R3.json","w"),default=float); print("ALL DONE",time.time()-t0,flush=True)
