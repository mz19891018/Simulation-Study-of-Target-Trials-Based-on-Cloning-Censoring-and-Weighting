import json, numpy as np
from full3 import *
d=impute(P,100); dz=Design3(d); G=7; md=d.mmae_day.values; st=d.status.values; tm=d.time.values
H=hazard_model(d,dz,G); S=np.cumprod(1-H,axis=1); WA,cA,WB,cB,_=weights(d,dz,G)
rows=[]
def blk(a,b):
    rA=at_risk_day(d,a)&(cA>=a); rB=at_risk_day(d,a)&(cB>=a)
    inA=lambda s: int(((cA>=tm)&(tm>=a)&(tm<=b)&(st==s)).sum()); inB=lambda s: int(((cB>=tm)&(tm>=a)&(tm<=b)&(st==s)).sum())
    newm=int(((md>=a)&(md<=min(b,G))&at_risk_day(d,a)).sum()) if a<=G else 0
    cenA=int(((cA>=a)&(cA<=b)&(md>G)&~((st>0)&(tm<=G))).sum()) if a<=G<=b else 0
    cenB=int(((md>=a+1)&(md<=b+1)&(md<=G)).sum()) if a<G else 0
    hz=float(H[at_risk_day(d,a)&(md>=a),a].mean()) if 1<=a<=G else None
    wa=WA[rA,a]; wb=WB[rB,a]
    return {"int":f"{a}" if a==b else f"{a}～{b}","A_n":int(rA.sum()),"A_mmae":newm,"A_reop":inA(1),"A_death":inA(2),"A_cens":cenA,
            "B_n":int(rB.sum()),"B_cens":cenB,"B_reop":inB(1),"B_death":inB(2),"hz":hz,
            "A_w":[float(np.median(wa)),float(wa.max())],"B_w":[float(np.median(wb)),float(wb.max())]}
for a,b in [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,30),(31,90),(91,180)]: rows.append(blk(a,b))
r8=at_risk_day(d,8); pA=float((1-S[r8&(md>G),G]).mean())
json.dump({"rows":rows,"pA_cens_mean":pA,"pA_all":float((1-S[r8,G]).mean())},open("riskset.json","w"),ensure_ascii=False,indent=1)
for r in rows: print(r)
print(pA)
