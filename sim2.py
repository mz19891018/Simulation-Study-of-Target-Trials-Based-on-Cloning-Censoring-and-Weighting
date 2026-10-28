import numpy as np, pandas as pd
rng=np.random.default_rng(23)
N=812; D=181
def tn(m,s,lo,hi,n):
    x=rng.normal(m,s,n)
    while ((x<lo)|(x>hi)).any():
        b=(x<lo)|(x>hi); x[b]=rng.normal(m,s,b.sum())
    return x
d=pd.DataFrame({"id":np.arange(1,N+1)})
d["center"]=rng.choice(["A","B"],N,p=[.58,.42])
d["team"]=np.where(d.center=="A",rng.choice(["A1","A2","A3"],N,p=[.4,.35,.25]),rng.choice(["B1","B2","B3"],N,p=[.45,.3,.25]))
d["year"]=rng.choice([2020,2021,2022,2023,2024,2025],N,p=[.13,.16,.19,.21,.22,.09])
d["age"]=np.round(tn(74.2,8.9,50,96,N)).astype(int)
d["male"]=rng.binomial(1,.75,N)
at=rng.choice(["SAPT","DAPT","DOAC","VKA","AP+AC"],N,p=[.46,.12,.23,.08,.11]); d["at_type"]=at
d["anticoag"]=np.isin(at,["DOAC","VKA","AP+AC"]).astype(int)
d["af"]=np.where(d.anticoag==1,rng.binomial(1,.85,N),rng.binomial(1,.04,N))
d["stent12"]=np.where(at=="DAPT",rng.binomial(1,.7,N),np.where(at=="AP+AC",rng.binomial(1,.35,N),rng.binomial(1,.05,N)))
d["prior_stroke"]=rng.binomial(1,.31,N)
d["mech_valve"]=np.where(at=="VKA",rng.binomial(1,.3,N),0)
d["htn"]=rng.binomial(1,.64,N); d["dm"]=rng.binomial(1,.27,N)
d["chf"]=rng.binomial(1,.12,N)
chads=d.chf+d.htn+2*(d.age>=75)+d.dm+2*d.prior_stroke+(d.age.between(65,74))+(1-d.male)
d["high_thromb"]=(((d.af==1)&(chads>=4))|(d.mech_valve==1)|(d.stent12==1)).astype(int)
d["alcohol"]=rng.binomial(1,np.where(d.male==1,.15,.03))
d["egfr_lt60"]=rng.binomial(1,np.clip(.18+.006*(d.age-70),.05,.6))
d["pre_mrs"]=rng.choice([0,1,2,3],N,p=[.45,.27,.18,.10])
d["bilateral"]=rng.binomial(1,.21,N)
d["thick"]=np.round(tn(22.5,5.2,10,40,N),1)
d["mls"]=np.round(np.clip(0.38*d.thick-0.6+rng.normal(0,2.6,N)-3.0*d.bilateral,0,22),1)
d["nakaguchi"]=rng.choice(["均质型","分层型","分隔型","小梁型"],N,p=[.31,.24,.30,.15])
sep=d.nakaguchi.isin(["分隔型","小梁型"]).astype(int)
d["hyperdense"]=rng.binomial(1,np.clip(.25+.15*sep,0,1))
d["markwalder"]=rng.choice([1,2,3,4],N,p=[.27,.52,.16,.05])
d["plt"]=np.round(tn(198,58,60,420,N)).astype(int)
d["inr"]=np.round(np.where(at=="VKA",tn(1.9,.5,1,4,N),np.where(np.isin(at,["DOAC","AP+AC"]),tn(1.18,.12,.9,1.8,N),tn(1.02,.07,.85,1.3,N))),2)
d["reversal"]=np.where(at=="VKA",rng.binomial(1,.7,N),np.where(np.isin(at,["DOAC","AP+AC"]),rng.binomial(1,.25,N),rng.binomial(1,.18,N)))
d["two_burr"]=rng.binomial(1,.34,N); d["subgaleal"]=rng.binomial(1,.2,N)
d["atorva"]=rng.binomial(1,np.clip(.25+.02*(d.year-2020),0,1))
# postoperative day-1 CT (time-varying confounder measured before MMAE decision)
d["resid_ml"]=np.round(np.exp(np.log(32)+0.035*(d.thick-22.5)+0.25*d.hyperdense+0.3*d.bilateral+0.2*sep-0.15*d.two_burr+rng.normal(0,.45,N)),1)
d["pneumo_ml"]=np.round(np.exp(np.log(12)+0.2*d.two_burr-0.25*d.subgaleal+rng.normal(0,.6,N)),1)
team_eff={"A1":.6,"A2":0.0,"A3":-.5,"B1":.3,"B2":-.2,"B3":-.7}
x_mmae=(-4.15+0.55*sep+0.45*d.bilateral+0.04*(d.thick-22.5)+0.012*(d.resid_ml-32)+0.35*d.hyperdense+0.35*d.anticoag
        +0.25*(d.year-2022)-0.03*(d.age-74)-0.45*d.egfr_lt60-0.25*(d.pre_mrs>=3)+d.team.map(team_eff))
x_res=(-3.25+0.75*d.high_thromb+0.25*d.anticoag+0.3*d.mech_valve-0.25*d.bilateral-0.004*(d.resid_ml-32))
x_rec=(0.40*d.anticoag+0.25*(at=="AP+AC")+0.55*sep+0.35*d.hyperdense+0.32*d.bilateral+0.030*(d.thick-22.5)+0.014*(d.resid_ml-32)
       +0.008*(d.pneumo_ml-12)+0.25*d.alcohol-0.25*d.atorva+0.25*(d.markwalder>=3)-0.15*d.two_burr-0.2*d.subgaleal-0.1*d.reversal)
x_die=0.07*(d.age-74)+0.5*(d.markwalder>=3)+0.4*d.egfr_lt60+0.35*(d.pre_mrs>=2)+0.25*d.chf
x_te=0.7*d.high_thromb+0.4*d.prior_stroke+0.3*d.af
x_pn=0.04*(d.age-74)+0.5*(d.pre_mrs>=2)+0.3*d.chf
mmae_day=np.full(N,999); res_day=np.full(N,999); ev=np.zeros(N,int); evday=np.full(N,180)
te_day=np.full(N,999); pn_day=np.full(N,999)
lost=np.where(rng.random(N)<0.04,rng.integers(15,180,N),999)
tshape=lambda t: np.where(t<7,0.25,np.where(t<60,1.0,np.where(t<120,0.6,0.35)))
sympt_cons=np.zeros(N,int); sr_day=np.full(N,999)
for t in range(0,181):
    alive=(ev==0)&(lost>t)
    # MMAE initiation days 1-7, rare later
    if 1<=t<=7: h=1/(1+np.exp(-(x_mmae+np.log(1+(t<=3)*1.2)+0.012*(np.where(t>=1,1,0)))))
    elif 8<=t<=30: h=1/(1+np.exp(-(x_mmae-3.8)))
    else: h=np.zeros(N)
    newm=alive&(mmae_day==999)&(rng.random(N)<h); mmae_day[newm]=t
    # antithrombotic resumption (mediator): MMAE increases hazard
    if t>=1:
        hr=1/(1+np.exp(-(x_res+0.55*(mmae_day<=t)+0.04*min(t,20))))
        newr=alive&(res_day==999)&(rng.random(N)<hr); res_day[newr]=t
    onat=res_day<=t; onm=mmae_day<=t
    early_ac=(res_day<=7)&(d.anticoag.values==1)
    hrm=np.where(onm,np.where(early_ac,0.78,0.40),1.0)
    lam=0.00056*np.exp(x_rec)*tshape(t)*hrm*np.where(onat,1.35+0.35*d.anticoag.values,1.0)
    r=alive&(rng.random(N)<lam)
    # some symptomatic recurrences managed conservatively (more often after MMAE)
    sr_day[r&(sr_day==999)]=t
    cons=r&(rng.random(N)<np.where(onm,0.14,0.08)); sympt_cons[cons]=1
    reop=r&~cons
    ev[reop]=1; evday[reop]=t
    dd=alive&~reop&(rng.random(N)<0.00030*np.exp(x_die)); ev[dd]=2; evday[dd]=t
    te=(te_day==999)&alive&(rng.random(N)<0.00028*np.exp(x_te)*np.where(onat,0.55,1.0)); te_day[te]=t
    pn=(pn_day==999)&alive&(t>=8)&(rng.random(N)<0.00030*np.exp(x_pn)); pn_day[pn]=t
cens=(ev==0)&(lost<180)
evday=np.where(cens,lost,evday)
d["mmae_day"]=mmae_day; d["res_day"]=res_day; d["status"]=ev; d["time"]=evday; d["lost"]=cens.astype(int)
d["sympt_cons"]=sympt_cons; d["sr_day"]=np.where(sr_day<=d.time,sr_day,999)
d["te"]=((te_day<=d.time)).astype(int); d["te_day"]=np.where(d.te==1,te_day,999)
d["pneu"]=((pn_day<=d.time)).astype(int); d["pn_day"]=np.where(d.pneu==1,pn_day,999)
# radiological recurrence at scheduled CTs (~30/90/180) — objective, partially detected
mm=(d.mmae_day<=7).astype(int)
prad=1/(1+np.exp(-(-1.0+0.9*(d.status==1)+0.35*sep+0.25*d.anticoag+0.3*d.bilateral+0.015*(d.resid_ml-32)-0.95*mm)))
d["rad"]=np.where(d.status==1,1,rng.binomial(1,prad))
d.loc[d.status==2,"rad"]=np.where(rng.random((d.status==2).sum())<.15,1,0)
d["n_ct"]=1+rng.poisson(np.clip(2.6+0.25*mm+1.2*(d.status==1)+0.6*d.sympt_cons,0,None))
# mRS at 180 d (ordinal 0-6)
lat=(0.55*d.pre_mrs+0.04*(d.age-74)+0.7*(d.markwalder>=3)+0.9*(d.status==1)+0.5*d.te-0.05*mm+rng.logistic(0,1,N)*0.9)
m=np.digitize(lat,[-0.6,0.5,1.4,2.3,3.4]); d["mrs180"]=np.where(d.status==2,6,m)
d["los"]=np.round(np.exp(rng.normal(np.log(9.5)+0.12*mm+0.15*(d.markwalder>=3),0.32))).astype(int)
d["agent"]=np.where(mm==1,rng.choice(["Onyx","NBCA","颗粒+弹簧圈"],N,p=[.6,.32,.08]),"")
for c,p in [("mls",.018),("plt",.02),("inr",.035),("pneumo_ml",.03)]:
    d.loc[rng.random(N)<p,c]=np.nan
d.to_csv("cohort2.csv",index=False)
print("MMAE<=7:",(d.mmae_day<=7).sum(),"late MMAE:",((d.mmae_day>7)&(d.mmae_day<999)).sum())
print("res<=7:",(d.res_day<=7).sum(),"res<=30",(d.res_day<=30).sum(),"never",(d.res_day==999).sum())
print(d.status.value_counts().to_dict(),"lost",d.lost.sum())
print(pd.crosstab(mm,d.status,normalize="index"))
print("early events<7", ((d.status>0)&(d.time<7)).sum(), "te",d.te.sum(),"pneu",d.pneu.sum(),"cons",d.sympt_cons.sum())
print("res<=7 by mmae", d.groupby(mm).apply(lambda g:(g.res_day<=7).mean()).to_dict())
