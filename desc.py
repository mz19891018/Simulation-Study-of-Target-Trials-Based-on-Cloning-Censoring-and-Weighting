import numpy as np, pandas as pd, json
from scipy import stats
from ccw import *
raw=pd.read_csv("cohort2.csv"); P=prep(raw)
g=(P.mmae_day<=7).astype(int); P["g"]=g
out={}
out["N"]=len(P); out["nA"]=int(g.sum()); out["nB"]=int((1-g).sum()); out["late_mmae"]=int(((P.mmae_day>7)&(P.mmae_day<999)).sum())
out["mmae_day_med"]=list(np.percentile(P.mmae_day[g==1],[50,25,75]))
out["agents"]=P.agent[g==1].value_counts().to_dict()
out["bilat_emb"]=int(((g==1)&(P.bilateral==1)).sum())
out["res7"]={k:int(((P.res_day<=7)&(g==k)).sum()) for k in (0,1)}
out["res_med"]={k:list(np.percentile(np.where(P.res_day[g==k]==999,np.nan,P.res_day[g==k])[~np.isnan(np.where(P.res_day[g==k]==999,np.nan,P.res_day[g==k]))],[50,25,75])) for k in (0,1)}
out["never_res"]=int((P.res_day==999).sum())
out["events"]={k:{s:int(((P.status==s)&(g==k)).sum()) for s in (1,2)} for k in (0,1)}
out["early_events"]=int(((P.status>0)&(P.time<=7)).sum())
out["lost"]={k:int(((P.lost==1)&(g==k)).sum()) for k in (0,1)}
out["cons"]={k:int(((P.sympt_cons==1)&(g==k)).sum()) for k in (0,1)}
out["sr"]={k:int(((P.sr_day<999)&(g==k)).sum()) for k in (0,1)}
out["nct"]={k:[float(P.n_ct[g==k].mean()),float(P.n_ct[g==k].std())] for k in (0,1)}
out["nct_p"]=float(stats.mannwhitneyu(P.n_ct[g==1],P.n_ct[g==0]).pvalue)
out["te"]={k:int((P.te[g==k]).sum()) for k in (0,1)}
out["pneu"]={k:int((P.pneu[g==k]).sum()) for k in (0,1)}
out["los"]={k:list(np.percentile(P.los[g==k],[50,25,75])) for k in (0,1)}; out["los_p"]=float(stats.mannwhitneyu(P.los[g==1],P.los[g==0]).pvalue)
out["age"]=[float(P.age.mean()),float(P.age.std())]; out["male"]=int(P.male.sum()); out["anticoag"]=int(P.anticoag.sum()); out["high_thromb"]=int(P.high_thromb.sum())
out["missing"]={c:int(raw[c].isna().sum()) for c in ["mls","plt","inr","pneumo_ml"]}
out["center"]=P.center.value_counts().to_dict()
# Table 2 with statistics
rows=[]
def cont(name,col,dec=1):
    a=raw[col][g==1].dropna(); b=raw[col][g==0].dropna(); t=stats.ttest_ind(a,b)
    s=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
    rows.append([name,f"{a.mean():.{dec}f}±{a.std():.{dec}f}",f"{b.mean():.{dec}f}±{b.std():.{dec}f}",f"t＝{t.statistic:.2f}",t.pvalue,s])
def skew(name,col,dec=1):
    a=raw[col][g==1].dropna(); b=raw[col][g==0].dropna(); u=stats.mannwhitneyu(a,b)
    z=stats.norm.isf(u.pvalue/2)*np.sign(a.median()-b.median())
    s=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
    q=lambda x: f"{x.median():.{dec}f}（{x.quantile(.25):.{dec}f}，{x.quantile(.75):.{dec}f}）"
    rows.append([name,q(a),q(b),f"Z＝{z:.2f}",u.pvalue,s])
def binr(name,mask):
    m=mask.astype(int); a=m[g==1]; b=m[g==0]
    tab=[[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]]; c=stats.chi2_contingency(tab,correction=False)
    pa,pb=a.mean(),b.mean(); s=(pa-pb)/np.sqrt((pa*(1-pa)+pb*(1-pb))/2)
    rows.append([name,f"{a.sum()}（{pa*100:.1f}）",f"{b.sum()}（{pb*100:.1f}）",f"χ²＝{c[0]:.2f}",c[1],s])
def multi(name,col,levels):
    tab=[[((raw[col]==l)&(g==k)).sum() for l in levels] for k in (1,0)]; c=stats.chi2_contingency(tab,correction=False)
    first=True
    for l in levels:
        m=(raw[col]==l).astype(int); a=m[g==1]; b=m[g==0]; pa,pb=a.mean(),b.mean(); s=(pa-pb)/np.sqrt((pa*(1-pa)+pb*(1-pb))/2)
        rows.append([(name if first else "　")+("：" if first else "")+levels[l] if isinstance(levels,dict) else ((name+"：") if first else "　")+str(l),
                     f"{a.sum()}（{pa*100:.1f}）",f"{b.sum()}（{pb*100:.1f}）",f"χ²＝{c[0]:.2f}" if first else "",c[1] if first else None,s]); first=False
cont("年龄（岁，x̄±s）","age"); binr("男性",raw.male==1)
multi("抗栓方案","at_type",{"SAPT":"单一抗血小板","DAPT":"双联抗血小板","DOAC":"直接口服抗凝药","VKA":"华法林","AP+AC":"抗血小板+抗凝"})
binr("高血栓风险ᵃ",raw.high_thromb==1); binr("心房颤动",raw.af==1); binr("既往缺血性卒中",raw.prior_stroke==1)
binr("高血压",raw.htn==1); binr("糖尿病",raw.dm==1); binr("心力衰竭",raw.chf==1); binr("长期饮酒",raw.alcohol==1)
binr("eGFR＜60 mL·min⁻¹·(1.73 m²)⁻¹",raw.egfr_lt60==1); binr("病前mRS≥2分",raw.pre_mrs>=2)
binr("双侧血肿",raw.bilateral==1); cont("最大血肿厚度（mm，x̄±s）","thick"); cont("中线移位（mm，x̄±s）","mls")
multi("Nakaguchi分型","nakaguchi",{"均质型":"均质型","分层型":"分层型","分隔型":"分隔型","小梁型":"小梁型"})
binr("高密度或混杂密度",raw.hyperdense==1); binr("Markwalder 3～4级",raw.markwalder>=3)
cont("血小板计数（×10⁹/L，x̄±s）","plt",0); cont("INR（x̄±s）","inr",2); binr("术前抗栓逆转或血小板输注",raw.reversal==1)
binr("双孔钻孔引流",raw.two_burr==1); binr("帽状腱膜下引流",raw.subgaleal==1); binr("术后阿托伐他汀",raw.atorva==1)
skew("术后第1天残余血肿量［mL，M（Q1，Q3）］","resid_ml"); skew("术后第1天颅内积气量［mL，M（Q1，Q3）］","pneumo_ml")
binr("中心A",raw.center=="A"); binr("2023—2025年手术",raw.year>=2023)
def fp(p):
    if p is None: return ""
    return "＜0.001" if p<0.001 else f"{p:.3f}"
out["table2"]=[[r[0],r[1],r[2],r[3],fp(r[4]),round(float(r[5]),2)] for r in rows]
# validation of AJ against lifelines (unweighted)
from lifelines import AalenJohansenFitter
d=impute(P,0); W=np.ones((len(d),T+1)); c=np.full(len(d),10**6)
mine=cif_weighted(d,W,c,"reop")[0][180]
aj=AalenJohansenFitter(calculate_variance=False).fit(np.where(d.status>0,d.time,d.time+0.5).astype(float),d.status.values,event_of_interest=1)
out["aj_check"]=[float(mine),float(aj.cumulative_density_.iloc[:,0].loc[:180.5].iloc[-1])]
# sample size: two proportions 15% vs 7.5%
za,zb=1.96,0.8416; p1,p2=.15,.075; pbar=(p1+p2)/2
n=(za*np.sqrt(2*pbar*(1-pbar))+zb*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p1-p2)**2
out["n_per_group"]=float(n)
json.dump(out,open("desc.json","w"),ensure_ascii=False,indent=1,default=float)
for k,v in out.items():
    if k!="table2": print(k,v)
for r in out["table2"]: print(r)
