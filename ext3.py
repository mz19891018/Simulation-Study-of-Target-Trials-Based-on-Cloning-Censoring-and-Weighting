"""在cohort2.csv基础上追加时间戳、计划CT随访、结局可评估性与判定一致性（独立随机流，不改动原有列）"""
import numpy as np, pandas as pd
rng=np.random.default_rng(2026)
d=pd.read_csv("cohort2.csv"); N=len(d)
# 首次术后CT与MMAE开始时间（距手术结束的小时数）
d["ct1_h"]=np.round(np.clip(rng.normal(20,4,N),10,32),1)
md=d.mmae_day.values
h=np.where(md<999,24*(md-1)+rng.uniform(20,34,N),np.nan)
d["mmae_h"]=np.round(h,1)
# 计划CT（约1、3、6个月），存活且仍在随访者按概率完成
tend=d.time.values; alive_at=lambda t: tend>=t
sick=(d.pre_mrs>=2).astype(int).values
for lab,t0,p in [("ct1m",30,.94),("ct3m",90,.89),("ct6m",180,.85)]:
    day=np.round(t0+rng.normal(0,4 if t0<180 else 6,N)).astype(int); day=np.minimum(day,180)
    ok=alive_at(day)&(rng.random(N)<p-0.05*sick)
    d[lab]=ok.astype(int); d[lab+"_day"]=np.where(ok,day,np.nan)
sched=d[["ct1m","ct3m","ct6m"]].sum(1)
d["n_ct_sched"]=sched; d["n_ct_unsched"]=np.clip(d.n_ct-1-sched,0,None)
# 影像学复发：再手术者计为1；其余需至少1次计划CT方可评估
assess=(d.status==1)|(sched>0)
d["rad_obs"]=assess.astype(int); d["rad"]=np.where(assess,d.rad,np.nan)
# 180 d mRS：失访者缺失
d["mrs_obs"]=(d.lost==0).astype(int); d["mrs180"]=np.where(d.lost==1,np.nan,d.mrs180)
# 判定一致性：症状性复发（两名神经外科医师）与影像学复发（两名神经影像医师）
sr=(d.sr_day<999).astype(int).values
r2=np.where(sr==1,rng.random(N)>0.05,rng.random(N)<0.006).astype(int); d["sr_rater2"]=r2
rv=np.nan_to_num(d.rad.values,nan=-1).astype(int)
rr2=np.where(rv==1,rng.random(N)>0.07,np.where(rv==0,rng.random(N)<0.05,-1)); d["rad_rater2"]=np.where(rv>=0,rr2,np.nan)
d.to_csv("cohort3.csv",index=False)
def kappa(a,b):
    a=np.asarray(a);b=np.asarray(b);po=(a==b).mean();pe=a.mean()*b.mean()+(1-a.mean())*(1-b.mean());return (po-pe)/(1-pe)
m=d.rad_obs==1
print("kappa_sr",round(kappa(sr,r2),3),"kappa_rad",round(kappa(d.rad[m],d.rad_rater2[m]),3))
print("ct1_h median",d.ct1_h.median(),d.ct1_h.quantile([.25,.75]).tolist())
d1=d.mmae_day==1; print("day1 MMAE",d1.sum(),"before CT",(d1&(d.mmae_h<d.ct1_h)).sum())
print("rad_obs",d.rad_obs.sum(),"mrs_obs",d.mrs_obs.sum())
