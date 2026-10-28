import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
T=180
BASE=["age","male","anticoag","dapt","apac","vka","af","high_thromb","prior_stroke","htn","dm","chf","alcohol","egfr_lt60","pre_mrs2",
      "bilateral","thick","mls","nk_lam","nk_sep","nk_tra","hyperdense","mw34","plt","inr","reversal","two_burr","subgaleal","atorva",
      "resid_ml","pneumo_ml","yr"]
TEAMS=["A2","A3","B1","B2","B3"]
SPL=["age","thick","resid_ml"]
def prep(d):
    d=d.copy()
    d["dapt"]=(d.at_type=="DAPT").astype(int); d["apac"]=(d.at_type=="AP+AC").astype(int); d["vka"]=(d.at_type=="VKA").astype(int)
    d["pre_mrs2"]=(d.pre_mrs>=2).astype(int); d["mw34"]=(d.markwalder>=3).astype(int); d["yr"]=d.year-2022
    for k,v in [("nk_lam","分层型"),("nk_sep","分隔型"),("nk_tra","小梁型")]: d[k]=(d.nakaguchi==v).astype(int)
    d["centerA"]=(d.center=="A").astype(int)
    d["sep"]=d.nakaguchi.isin(["分隔型","小梁型"]).astype(int); d["age75"]=(d.age>=75).astype(int)
    for t in TEAMS: d["team_"+t]=(d.team==t).astype(int)
    for y in [2021,2022,2023,2024,2025]: d["y%d"%y]=(d.year==y).astype(int)
    return d
def impute(d,seed):
    imp=IterativeImputer(sample_posterior=True,random_state=seed,max_iter=10)
    d=d.copy(); d[BASE]=imp.fit_transform(d[BASE].values.astype(float)); return d
KEY=["age","thick","resid_ml","sep","bilateral","hyperdense","anticoag","high_thromb","egfr_lt60","pre_mrs2","mw34","reversal","pneumo_ml","yr","centerA"]
class Design:
    def __init__(self,d,exclude=(),reduced=False):
        self.SPL=[c for c in SPL if c not in exclude]
        self.spl=SplineTransformer(n_knots=4,degree=3).fit(d[self.SPL].values)
        if reduced:
            other=[c for c in KEY if c not in SPL+list(exclude)]
        else:
            other=[c for c in BASE if c not in SPL+["yr"]+list(exclude)]+["team_"+t for t in TEAMS]+["y%d"%y for y in [2021,2022,2023,2024,2025]]
        other=[c for c in other if d[c].nunique()>1]
        self.other=other; self.sc=StandardScaler().fit(self.raw(d))
    def raw(self,d): return np.hstack([self.spl.transform(d[self.SPL].values),d[self.other].values.astype(float)])
    def X(self,d): return self.sc.transform(self.raw(d))
def fit(X,y):
    m=LogisticRegression(C=100.0,max_iter=5000); m.fit(X,y); return m
def at_risk_day(d,t):
    # at risk (event-free, under follow-up) at START of day t (before initiation)
    return ((d.status.values>0)&(d.time.values>=t))|((d.status.values==0)&((d.time.values>t)|(d.time.values==T)))
def daily_hazard(d,dz,col,G,extra=None):
    """pooled logistic hazard of initiating `col` on days 1..G; returns array N x (G+1) (day0 = 0)"""
    X=dz.X(d); rows=[];ys=[];tt=[];ex=[]
    for t in range(1,G+1):
        r=at_risk_day(d,t)&(d[col].values>=t)
        rows.append(np.where(r)[0]); ys.append((d[col].values[r]==t).astype(int)); tt.append(np.full(r.sum(),t))
        if extra is not None: ex.append(extra(d,t)[r])
    idx=np.concatenate(rows); y=np.concatenate(ys); tv=np.concatenate(tt)
    def F(Xi,tv,e=None):
        cols=[Xi,(tv<=3)[:,None].astype(float),((tv-1)/G)[:,None]]
        if e is not None: cols.append(e[:,None])
        return np.hstack(cols)
    m=fit(F(X[idx],tv,np.concatenate(ex) if extra is not None else None),y)
    H=np.zeros((len(d),G+1))
    for t in range(1,G+1):
        e=extra(d,t) if extra is not None else None
        H[:,t]=m.predict_proba(F(X,np.full(len(d),t),e))[:,1]
    return H
def p_by_G(d,dz,col,G,extra_cols=None):
    """P(initiated col by end of day G | X, event-free & followed to day G)"""
    X=dz.X(d); r=at_risk_day(d,G+1)
    Xi=X if extra_cols is None else np.hstack([X,d[extra_cols].values.astype(float)])
    y=(d[col].values<=G).astype(int)
    m=fit(Xi[r],y[r]); return m.predict_proba(Xi)[:,1]
from scipy.optimize import minimize
LAM=0.005
def ebal(Xt,target,lam=None):
    lam=LAM if lam is None else lam
    Z=Xt-target
    def f(l):
        a=Z@l; m=a.max(); return m+np.log(np.exp(a-m).sum())+lam/2*l@l
    def g(l):
        a=Z@l; w=np.exp(a-a.max()); w/=w.sum(); return w@Z+lam*l
    r=minimize(f,np.zeros(Z.shape[1]),jac=g,method="BFGS",options={"maxiter":5000})
    a=Z@r.x; w=np.exp(a-a.max()); return w/w.sum()
def clone_weights(d,dz,strat,G=7,trunc=99,eb=True):
    """strat: dict col->{1: initiate by G, 0: not by G}. returns (W: N x (T+1) weights, cens_day: N, arrays)"""
    N=len(d); W=np.ones((N,T+1)); cens=np.full(N,10**6)
    for col,s in strat.items():
        day=d[col].values
        extra=None; extra_cols=None
        if col=="res_day":
            extra=lambda dd,t: (dd["mmae_day"].values<t).astype(float)
            d=d.assign(_m=(d.mmae_day.values<=G).astype(float)); extra_cols=["_m"]
        if s==1:
            dev=(day>G)
            cens=np.where(dev,np.minimum(cens,G),cens)  # censored after day G
            if eb and col=="mmae_day" and len(strat)==1:
                Xr=np.hstack([d[dz.SPL].values.astype(float),d[dz.other].values.astype(float)])
                X=(Xr-Xr.mean(0))/np.where(Xr.std(0)>0,Xr.std(0),1); r=at_risk_day(d,G+1); tr=r&~dev
                we=ebal(X[tr],X[r].mean(0))*r.sum()
                w=np.ones(len(d)); w[tr]=we
            else:
                p=p_by_G(d,dz,col,G,extra_cols)
                w=np.where(dev,1.0,1/np.clip(p,1e-3,1))
            W[:,G+1:]*=w[:,None]
        else:
            H=daily_hazard(d,dz,col,G,extra)
            dev=(day<=G)
            cens=np.where(dev,np.minimum(cens,day-1),cens)  # not at risk from initiation day
            surv=np.cumprod(1-H,axis=1)  # P(not initiated through day t)
            for t in range(T+1):
                W[:,t]*=1/np.clip(surv[:,min(t,G)],1e-3,1)
    # truncation at percentile among uncensored weights after grace
    if trunc is not None and not (eb and len(strat)==1 and list(strat.values())[0]==1):
        ok=cens>G; cap=np.percentile(W[ok,G+1],trunc) if ok.any() else np.inf
        W=np.minimum(W,cap)
    return W,cens
def cif_weighted(d,W,cens,outcome="reop",grid=(30,90,180)):
    st=d.status.values.copy(); tm=d.time.values
    if outcome=="any":
        st=st.copy()
    N=len(d); S=1.0; F1=0.0; F2=0.0; out={}; curve=np.zeros(T+1); rmtl=0.0
    for t in range(T+1):
        risk=at_risk_day(d,t)&(cens>=t)
        if outcome=="te": risk&=(d.te_day.values>=t)
        if outcome=="pneu": risk&=(d.pn_day.values>=t)
        if outcome=="any": risk&=(d.sr_day.values>=t)
        w=W[:,t]*risk
        if outcome=="any":
            e1=w*(d.sr_day.values==t); e2=w*((st==2)&(tm==t)&(d.sr_day.values!=t))
        elif outcome in ("reop",):
            e1=w*((st==1)&(tm==t)); e2=w*((st==2)&(tm==t))
        elif outcome=="death":
            e1=w*((st==2)&(tm==t)); e2=w*((st==1)&(tm==t))
        elif outcome=="te":
            td=d.te_day.values; e1=w*((td==t)); e2=w*((st>0)&(tm==t)&(td!=t))  # TE: competing = death/reop? use death only
            e2=w*((st==2)&(tm==t)&(td!=t))
        elif outcome=="pneu":
            pdays=d.pn_day.values if "pn_day" in d else None
            e1=w*(d.pn_day.values==t); e2=w*((st==2)&(tm==t)&(d.pn_day.values!=t))
        den=w.sum()
        if den>0:
            h1=e1.sum()/den; h2=e2.sum()/den
            F1+=S*h1; F2+=S*h2; S*=(1-h1-h2)
        curve[t]=F1
    for g in grid: out[g]=curve[g]
    out["rmtl"]=curve[:T].sum()  # days lost to outcome over 0-180
    return out,curve

def final_weights(d,W,cens,G=7):
    tm=d.time.values; k=np.minimum(tm,G+1).astype(int)
    inc=cens>=k
    w=W[np.arange(len(d)),k]*inc
    return w
