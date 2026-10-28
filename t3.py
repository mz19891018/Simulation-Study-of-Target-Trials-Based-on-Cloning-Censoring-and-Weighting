import numpy as np, pandas as pd, json
from scipy import stats
from full3 import *
g0=(P.mmae_day<=7).values
WS=[]
for m in range(5):
    d=impute(P,100+m); dz=Design3(d); WA,cA,WB,cB,_=weights(d,dz)
    WS.append((d,day8_weight(d,WA,cA),day8_weight(d,WB,cB)))
def smd_w(f):
    v=[]
    for d,wa,wb in WS:
        x=np.asarray(f(d),float); g=(d.mmae_day<=7).values; sd=np.sqrt((x[g].var()+x[~g].var())/2); v.append((np.average(x,weights=wa)-np.average(x,weights=wb))/sd)
    return float(np.mean(v))
def row(name,f,kind):
    x=np.asarray(f(P),float); ok=~np.isnan(x); a=x[g0&ok]; b=x[~g0&ok]
    sd=np.sqrt((a.var()+b.var())/2); s0=(a.mean()-b.mean())/sd
    if kind=="bin":
        ca,cb=int(a.sum()),int(b.sum()); chi=stats.chi2_contingency([[ca,len(a)-ca],[cb,len(b)-cb]],correction=False); st=f"χ²＝{chi[0]:.2f}"; p=chi[1]
        A=f"{ca}（{ca/len(a)*100:.1f}）"; B=f"{cb}（{cb/len(b)*100:.1f}）"
    elif kind=="norm":
        t=stats.ttest_ind(a,b); st=f"t＝{t.statistic:.2f}"; p=t.pvalue; dec=0 if a.mean()>100 else (2 if a.mean()<2 else 1)
        A=f"{a.mean():.{dec}f}±{a.std(ddof=1):.{dec}f}"; B=f"{b.mean():.{dec}f}±{b.std(ddof=1):.{dec}f}"
    else:
        u=stats.mannwhitneyu(a,b); z=stats.norm.isf(u.pvalue/2)*np.sign(np.median(a)-np.median(b)); st=f"Z＝{z:.2f}"; p=u.pvalue
        q=lambda v: f"{np.median(v):.1f}（{np.percentile(v,25):.1f}，{np.percentile(v,75):.1f}）"; A=q(a); B=q(b)
    ps="＜0.001" if p<0.001 else f"{p:.3f}"
    return f"| {name} | {A} | {B} | {st} | {ps} | {s0:.2f} | {smd_w(f):.2f} |".replace("| -0.00 |","| 0.00 |").replace("-","−")
MAIN=[("年龄（岁，x̄±s）",lambda d:d.age,"norm"),("男性",lambda d:d.male,"bin"),("含抗凝药物方案",lambda d:d.anticoag,"bin"),("　其中抗血小板联合抗凝",lambda d:d.apac,"bin"),
("高血栓风险ᵃ",lambda d:d.high_thromb,"bin"),("病前mRS≥2分",lambda d:d.pre_mrs2,"bin"),("eGFR＜60 mL·min⁻¹·(1.73 m²)⁻¹",lambda d:d.egfr_lt60,"bin"),("Markwalder 3～4级",lambda d:d.mw34,"bin"),
("双侧血肿",lambda d:d.bilateral,"bin"),("最大血肿厚度（mm，x̄±s）",lambda d:d.thick,"norm"),("中线移位（mm，x̄±s）",lambda d:d.mls,"norm"),("分隔型或小梁型（Nakaguchi）",lambda d:d.sep,"bin"),
("高密度或混杂密度",lambda d:d.hyperdense,"bin"),
("中心A",lambda d:d.centerA,"bin"),("2023—2025年手术",lambda d:(d.year>=2023).astype(int),"bin")]
CTR=[("残余血肿量［mL，M（Q1，Q3）］",lambda d:d.resid_ml,"skew"),("颅内积气量［mL，M（Q1，Q3）］",lambda d:d.pneumo_ml,"skew")]
EXTRA=[("术前抗栓逆转或血小板输注",lambda d:d.reversal,"bin"),("双孔钻孔",lambda d:d.two_burr,"bin"),("帽状腱膜下引流",lambda d:d.subgaleal,"bin"),("单一抗血小板",lambda d:(d.at_type=="SAPT").astype(int),"bin"),("双联抗血小板",lambda d:d.dapt,"bin"),("直接口服抗凝药",lambda d:(d.at_type=="DOAC").astype(int),"bin"),("华法林",lambda d:d.vka,"bin"),
("心房颤动",lambda d:d.af,"bin"),("既往缺血性卒中",lambda d:d.prior_stroke,"bin"),("高血压",lambda d:d.htn,"bin"),("糖尿病",lambda d:d.dm,"bin"),("心力衰竭",lambda d:d.chf,"bin"),("长期饮酒",lambda d:d.alcohol,"bin"),
("Nakaguchi均质型",lambda d:(d.nakaguchi=="均质型").astype(int),"bin"),("Nakaguchi分层型",lambda d:d.nk_lam,"bin"),("Nakaguchi分隔型",lambda d:d.nk_sep,"bin"),("Nakaguchi小梁型",lambda d:d.nk_tra,"bin"),
("血小板计数（×10⁹/L，x̄±s）",lambda d:d.plt,"norm"),("INR（x̄±s）",lambda d:d.inr,"norm"),("术后阿托伐他汀",lambda d:d.atorva,"bin")]
H="| 变量 | 7 d内行MMAE（n＝275） | 7 d内未行MMAE（n＝537） | 统计量 | P值 | 加权前SMD | 加权后SMD |\n| --- | --- | --- | --- | --- | --- | --- |"
main=[H]+[row(*r) for r in MAIN]+["| **术后第1天CT（时间零点后、第2天起进入删失模型）** |  |  |  |  |  |  |"]+[row(*r) for r in CTR]
ext=[H]+[row(*r) for r in EXTRA]
open("t3_main.md","w").write("\n".join(main)); open("t3_extra.md","w").write("\n".join(ext))
print("\n".join(main)); print("\n".join(ext))
