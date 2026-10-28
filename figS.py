import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":["Noto Sans CJK SC","DejaVu Sans"],"font.size":9,"axes.unicode_minus":False})
from full3 import *
d=impute(P,100); dz=Design3(d); G=7; md=d.mmae_day.values
H=hazard_model(d,dz,G); S=np.cumprod(1-H,axis=1); WA,cA,WB,cB,_=weights(d,dz,G)
NAV="#1F4E79"; GR="#7F7F7F"; RED="#B03A2E"
fig,ax=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={"width_ratios":[1.4,1]})
data=[]; labs=[]
for t in range(1,8):
    r=at_risk_day(d,t)&(cB>=t); data.append(WB[r,t]); labs.append(f"B\n第{t}天")
r8=at_risk_day(d,8); data.append(WA[r8&(md<=G),8]); labs.append("A\n第8天起")
bp=ax[0].boxplot(data,whis=(1,99),showfliers=True,flierprops=dict(marker=".",ms=3,mec=GR),patch_artist=True,medianprops=dict(color="k"))
for i,b in enumerate(bp["boxes"]): b.set_facecolor("#DCE6F1" if i<7 else "#F6DDD9")
ax[0].set_yscale("log"); ax[0].set_xticklabels(labs,fontsize=7.5); ax[0].set_ylabel("克隆权重（对数刻度）")
ax[0].set_title("a　逐日累积权重分布（须为第1～99百分位数）",fontsize=9,loc="left")
p7=1-S[:,G]; g=md<=G; bins=np.linspace(0,1,26)
ax[1].hist(p7[r8&~g],bins=bins,color=GR,alpha=.6,label="7 d内未行MMAE"); ax[1].hist(p7[r8&g],bins=bins,color=NAV,alpha=.7,label="7 d内行MMAE")
ax[1].set_xlabel("预测7 d内行MMAE的概率"); ax[1].set_ylabel("例数"); ax[1].legend(frameon=False,fontsize=8)
ax[1].set_title("b　第8天风险集中的重叠情况",fontsize=9,loc="left")
for a in ax: a.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fS_weights.png",dpi=300)
