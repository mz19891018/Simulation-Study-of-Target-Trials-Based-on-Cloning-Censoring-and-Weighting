"""v3：时间有序的逐日逆概率删失加权（两策略共用同一MMAE日风险模型）"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from ccw import prep, impute, BASE, TEAMS, KEY, fit, at_risk_day, ebal, cif_weighted, T
from sklearn.preprocessing import SplineTransformer, StandardScaler
CT=["resid_ml","pneumo_ml"]
CPEN=1.0
class Design3:
    """基线部分 Xb 与术后第1天CT部分 Xc 分开，便于按信息可得时间进入模型"""
    def __init__(self,d,reduced=False,no_ct=False):
        self.no_ct=no_ct
        self.sb=SplineTransformer(n_knots=4,degree=3).fit(d[["age","thick"]].values)
        self.sc_=SplineTransformer(n_knots=4,degree=3).fit(d[["resid_ml"]].values)
        if reduced: other=[c for c in KEY if c not in ["age","thick","resid_ml","pneumo_ml"]]
        else: other=[c for c in BASE if c not in ["age","thick","resid_ml","pneumo_ml","yr"]]+["team_"+t for t in TEAMS]+["y%d"%y for y in [2021,2022,2023,2024,2025]]
        self.other=[c for c in other if c in d and d[c].nunique()>1]
        self.Sb=StandardScaler().fit(self._b(d)); self.Sc=StandardScaler().fit(self._c(d))
    def _b(self,d): return np.hstack([self.sb.transform(d[["age","thick"]].values),d[self.other].values.astype(float)])
    def _c(self,d): return np.hstack([self.sc_.transform(d[["resid_ml"]].values),d[["pneumo_ml"]].values.astype(float)])
    def Xb(self,d): return self.Sb.transform(self._b(d))
    def Xc(self,d): return self.Sc.transform(self._c(d))
def hazard_model(d,dz,G=7,ct_day1=False):
    """逐日池化Logistic：第t天（1..G）开始时仍未栓塞且无事件者，当日开始MMAE的条件概率。
    CT变量仅在其已可得的决策日进入：主分析为第2天起；ct_day1=True时第1天亦纳入（需另行排除MMAE早于CT者）"""
    Xb=dz.Xb(d); Xc=dz.Xc(d) if not dz.no_ct else np.zeros((len(d),0))
    def F(idx,t):
        a=np.full(len(idx),1.0 if (t>=2 or ct_day1) else 0.0)[:,None]
        dayf=np.zeros((len(idx),G)); dayf[:,t-1]=1   # 每日独立截距
        return np.hstack([Xb[idx],Xc[idx]*a,dayf])
    rows=[];ys=[]
    for t in range(1,G+1):
        r=np.where(at_risk_day(d,t)&(d.mmae_day.values>=t))[0]
        rows.append(F(r,t)); ys.append((d.mmae_day.values[r]==t).astype(int))
    from sklearn.linear_model import LogisticRegression
    m=LogisticRegression(C=CPEN,max_iter=5000,fit_intercept=False).fit(np.vstack(rows),np.concatenate(ys))
    H=np.zeros((len(d),G+1)); allidx=np.arange(len(d))
    for t in range(1,G+1): H[:,t]=m.predict_proba(F(allidx,t))[:,1]
    return H
def ebal_base(Z,logq,lam=0.005):
    from scipy.optimize import minimize
    def f(l):
        a=logq+Z@l; m=a.max(); return m+np.log(np.exp(a-m).sum())+lam/2*l@l
    def g(l):
        a=logq+Z@l; w=np.exp(a-a.max()); w/=w.sum(); return w@Z+lam*l
    r=minimize(f,np.zeros(Z.shape[1]),jac=g,method="BFGS",options={"maxiter":5000})
    a=logq+Z@r.x; w=np.exp(a-a.max()); return w/w.sum()
def weights(d,dz,G=7,ct_day1=False,trunc=None,eb_A=False,calib=True):
    """返回 WA,cA,WB,cB,S（S[:,t]=P(第1..t天均未栓塞|X)）"""
    N=len(d); md=d.mmae_day.values
    H=hazard_model(d,dz,G,ct_day1); S=np.cumprod(1-H,axis=1)
    # 策略A：宽限期内不删失（权重1）；第G天末仍未栓塞者人工删失；第G+1天起已栓塞者权重=1/P(G天内栓塞)
    cA=np.where(md>G,G,10**6); WA=np.ones((N,T+1))
    pA=1-S[:,G]
    wA=np.where(md<=G,1/np.clip(pA,1e-3,1),0.0)
    if eb_A:
        Xr=np.hstack([dz.Xb(d),dz.Xc(d)]); r=at_risk_day(d,G+1); tr=r&(md<=G)
        wA=np.zeros(N); wA[tr]=ebal(Xr[tr],Xr[r].mean(0))*r.sum()
    # 策略B：栓塞当日前一天末人工删失；第t天权重=1/P(第1..min(t,G)天均未栓塞)
    cB=np.where(md<=G,md-1,10**6); WB=1/np.clip(S[:,np.minimum(np.arange(T+1),G)],1e-3,1)
    if calib and not eb_A:
        # 以基线协变量及术后第1天CT（第8天前均已测得）将第G+1天两策略的权重校准至第G+1天全体风险集（基础权重为逐日IPCW）
        Xb=np.hstack([dz.Xb(d),dz.Xc(d)]) if not dz.no_ct else dz.Xb(d); r=at_risk_day(d,G+1); tgt=Xb[r].mean(0)
        trA=r&(md<=G); wA_new=np.zeros(N); wA_new[trA]=ebal_base(Xb[trA]-tgt,np.log(wA[trA]))*r.sum(); wA=wA_new
        trB=r&(md>G); qB=WB[trB,G]; cal=ebal_base(Xb[trB]-tgt,np.log(qB))*r.sum()
        fB=np.ones(N); fB[trB]=cal/qB; WB=WB.copy(); WB[:,G+1:]*=fB[:,None]
    if trunc is not None:
        okA=(md<=G)&at_risk_day(d,G+1); capA=np.percentile(wA[okA],trunc); wA=np.minimum(wA,capA)
        okB=(md>G)&at_risk_day(d,G+1); capB=np.percentile(WB[okB,G],trunc); WB=np.minimum(WB,capB)
    WA[:,G+1:]=wA[:,None]
    return WA,cA,WB,cB,S
def obs_weight(d,dz,obs_col):
    """结局可观测概率倒数（logistic：全部协变量＋7 d内MMAE）"""
    X=np.hstack([dz.Xb(d),dz.Xc(d),(d.mmae_day.values<=7)[:,None].astype(float)])
    y=d[obs_col].values.astype(int)
    if y.min()==1: return np.ones(len(d))
    m=fit(X,y); p=m.predict_proba(X)[:,1]; return np.where(y==1,1/np.clip(p,0.05,1),0.0)
def day8_weight(d,W,c,G=7):
    k=np.minimum(d.time.values,G+1).astype(int); inc=c>=k
    return W[np.arange(len(d)),k]*inc

def res_hazard(d,dz,G=7):
    """抗栓恢复日风险模型（联合策略用）：含当日是否已行MMAE（随时间变化）"""
    Xb=dz.Xb(d); Xc=dz.Xc(d)
    def F(idx,t):
        dayf=np.zeros((len(idx),G)); dayf[:,t-1]=1
        return np.hstack([Xb[idx],Xc[idx],(d.mmae_day.values[idx]<t)[:,None].astype(float),dayf])
    rows=[];ys=[]
    for t in range(1,G+1):
        r=np.where(at_risk_day(d,t)&(d.res_day.values>=t))[0]; rows.append(F(r,t)); ys.append((d.res_day.values[r]==t).astype(int))
    from sklearn.linear_model import LogisticRegression
    m=LogisticRegression(C=CPEN,max_iter=5000,fit_intercept=False).fit(np.vstack(rows),np.concatenate(ys))
    H=np.zeros((len(d),G+1)); a=np.arange(len(d))
    for t in range(1,G+1): H[:,t]=m.predict_proba(F(a,t))[:,1]
    return H
def seq_w(day,S,s,G=7):
    """单一维度策略的克隆权重与删失日。s=1：G天内开始；s=0：G天内不开始"""
    N=len(day); W=np.ones((N,T+1))
    if s==1:
        c=np.where(day>G,G,10**6); w=np.where(day<=G,1/np.clip(1-S[:,G],1e-3,1),0.0); W[:,G+1:]=w[:,None]
    else:
        c=np.where(day<=G,day-1,10**6); W=1/np.clip(S[:,np.minimum(np.arange(T+1),G)],1e-3,1)
    return W,c
