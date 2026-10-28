"""补充材料诊断：逐日风险集、删失概率、权重分布、极端权重影响、重叠、CT随访与判定一致性"""
import json, numpy as np, pandas as pd
from full3 import *
d=impute(P,100); dz=Design3(d); G=7; md=d.mmae_day.values
H=hazard_model(d,dz,G); S=np.cumprod(1-H,axis=1)
WA,cA,WB,cB,_=weights(d,dz,G)
out={}
# 1 逐日风险集
rows=[]
for t in [0,1,2,3,4,5,6,7,8,14,30,90,180]:
    rA=at_risk_day(d,t)&(cA>=t); rB=at_risk_day(d,t)&(cB>=t)
    ev=lambda r,s: int((r&(d.status.values==s)&(d.time.values==t)).sum())
    newm=int((rA&(md==t)).sum()) if 1<=t<=G else 0
    censA=int((rA&(cA==t)).sum()); censB=int((at_risk_day(d,t+1)&(cB==t)).sum()) if t<G else 0
    # 策略B当日人工删失概率：仍未栓塞者当日开始MMAE的预测概率均值
    hB=float(H[rB&(md>=t+1) if False else rB,t+1].mean()) if t<G else 0.0
    wB=WB[rB,t]; wA=WA[rA,t]
    rows.append({"day":t,"A_n":int(rA.sum()),"A_newMMAE":newm,"A_reop":ev(rA,1),"A_death":ev(rA,2),"A_cens":censA,
                 "B_n":int(rB.sum()),"B_reop":ev(rB,1),"B_death":ev(rB,2),"B_cens":censB,"B_hz_next":round(hB,4),
                 "A_w_med":float(np.median(wA)),"A_w_max":float(wA.max()),"B_w_med":float(np.median(wB)),"B_w_p99":float(np.percentile(wB,99)),"B_w_max":float(wB.max())})
out["riskset"]=rows
# 2 第8天权重
r8=at_risk_day(d,8); wa=WA[r8&(md<=G),8]; wb=WB[r8&(md>G),8]
def desc(w): return {"n":int(len(w)),"min":float(w.min()),"q1":float(np.percentile(w,25)),"med":float(np.median(w)),"q3":float(np.percentile(w,75)),"p99":float(np.percentile(w,99)),"max":float(w.max()),"sum":float(w.sum()),"ess":float(w.sum()**2/(w**2).sum()),"top1pct_share":float(np.sort(w)[::-1][:max(1,int(len(w)*.01))].sum()/w.sum()),"top10_share":float(np.sort(w)[::-1][:10].sum()/w.sum())}
out["wA"]=desc(wa); out["wB"]=desc(wb)
out["pUncensA_mean"]=float((1-S[r8,G]).mean())
# 3 极端权重：剔除策略A权重最高的k例后重估
def rd_without(ids):
    dd=d[~d.id.isin(ids)].reset_index(drop=True); z=Design3(dd); a,ca,b,cb,_=weights(dd,z,G)
    return cif_weighted(dd,a,ca,"reop")[0][180]-cif_weighted(dd,b,cb,"reop")[0][180]
order=d.id.values[r8&(md<=G)][np.argsort(-wa)]
base=cif_weighted(d,WA,cA,"reop")[0][180]-cif_weighted(d,WB,cB,"reop")[0][180]
out["influence"]={"base":float(base),"drop5":float(rd_without(order[:5])),"drop10":float(rd_without(order[:10]))}
top=d[d.id.isin(order[:10])]; out["top10_events"]=int((top.status==1).sum())
# 4 重叠与中心/年度支持
p7=1-S[:,G]; g=md<=G
out["overlap"]={"treated":[float(np.percentile(p7[r8&g],q)) for q in (0,1,50,99,100)],"untreated":[float(np.percentile(p7[r8&~g],q)) for q in (0,1,50,99,100)],
                "n_p_lt_005":int((r8&(p7<0.05)).sum()),"n_p_lt_005_treated":int((r8&g&(p7<0.05)).sum())}
cy=[]
for (c,y),gg in d.assign(p7=p7,g=g).groupby(["center","year"]):
    cy.append({"center":c,"year":int(y),"n":len(gg),"mmae":int(gg.g.sum()),"p7_med":float(gg.p7.median())})
out["center_year"]=cy
# 5 均衡：第8天两策略加权人群相对目标（第8天全体风险集）
bal=[]
for col in KEYBAL:
    x=d[col].values.astype(float); sd=np.sqrt((x[g].var()+x[~g].var())/2)
    mA=np.average(x,weights=WA[:,8]*r8*(cA>=8)); mB=np.average(x,weights=WB[:,8]*r8*(cB>=8)); m0=x[r8].mean()
    bal.append((col,(mA-mB)/sd,(mA-m0)/sd,(mB-m0)/sd))
out["balance8"]=bal
# 6 CT与判定
R=raw
mm=(R.mmae_day<=7)
ct={}
for lab,t0 in [("ct1m",30),("ct3m",90),("ct6m",180)]:
    elig=(R.time>=t0-8)
    ct[lab]={"A":[int(R[lab][elig&mm].sum()),int((elig&mm).sum())],"B":[int(R[lab][elig&~mm].sum()),int((elig&~mm).sum())],
             "day_med":[float(R[lab+"_day"][mm].median()),float(R[lab+"_day"][~mm].median())]}
ct["unsched"]=[float(R.n_ct_unsched[mm].mean()),float(R.n_ct_unsched[~mm].mean())]
from scipy.stats import mannwhitneyu
ct["unsched_p"]=float(mannwhitneyu(R.n_ct_unsched[mm],R.n_ct_unsched[~mm]).pvalue)
ct["sched_mean"]=[float(R.n_ct_sched[mm].mean()),float(R.n_ct_sched[~mm].mean())]
ct["ct1_h"]=[float(R.ct1_h.median()),float(R.ct1_h.quantile(.25)),float(R.ct1_h.quantile(.75))]
ct["day1_mmae"]=int((R.mmae_day==1).sum()); ct["preCT"]=int(((R.mmae_day==1)&(R.mmae_h<R.ct1_h)).sum())
ct["rad_obs"]=[int(R.rad_obs[mm].sum()),int(mm.sum()),int(R.rad_obs[~mm].sum()),int((~mm).sum())]
ct["mrs_obs"]=[int(R.mrs_obs[mm].sum()),int(mm.sum()),int(R.mrs_obs[~mm].sum()),int((~mm).sum())]
def kappa(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);po=(a==b).mean();pe=a.mean()*b.mean()+(1-a.mean())*(1-b.mean());return (po-pe)/(1-pe)
ct["kappa_sr"]=float(kappa((R.sr_day<999).astype(int),R.sr_rater2)); m=R.rad_obs==1; ct["kappa_rad"]=float(kappa(R.rad[m],R.rad_rater2[m]))
out["ct"]=ct
# 联合策略正值性：按实际情况遵循各联合策略的人数（宽限期末）
J={}
rs=d.res_day.values
for k,(a,b) in {"MR":(1,1),"Mr":(1,0),"mR":(0,1),"mr":(0,0)}.items():
    f=r8&((md<=G)==bool(a))&((rs<=G)==bool(b)); J[k]=int(f.sum())
out["joint_follow"]=J
json.dump(out,open("diag3.json","w"),ensure_ascii=False,indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if k not in("riskset","center_year","balance8")},ensure_ascii=False,default=float)[:3000])
for r in rows: print(r)
print(sorted([(b[0],round(b[1],3)) for b in bal],key=lambda z:-abs(z[1]))[:8])
print([c for c in cy if c["mmae"]<3 or c["p7_med"]<0.1])
