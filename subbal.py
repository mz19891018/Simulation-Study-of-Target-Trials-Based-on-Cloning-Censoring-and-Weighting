import json, numpy as np
from full3 import *
KEYB=[c for c in KEY if c not in ("yr",)]+["yr"]
out={}
def keybal(o):
    d=data_for({**o,"mseed":100},None); dz=Design3(d,reduced=True)
    WA,cA,WB,cB,_=weights(d,dz); wa=day8_weight(d,WA,cA); wb=day8_weight(d,WB,cB)
    r8=at_risk_day(d,8); g=(d.mmae_day<=7).values; m=0
    for col in KEYB:
        x=d[col].values.astype(float); sd=np.sqrt((x[g].var()+x[~g].var())/2)
        if sd>0: m=max(m,abs(np.average(x,weights=wa*r8)-np.average(x,weights=wb*r8))/sd)
    return m
for col in ("anticoag","high_thromb","sep","age75"):
    for l in (0,1): out[f"{col}{l}"]=keybal({"subgroup":(col,l)})
for f in ("centerA","centerB"): out[f]=keybal({"filter":f})
out["reduced"]=keybal({})
json.dump(out,open("subbal.json","w")); print({k:round(v,3) for k,v in out.items()})
