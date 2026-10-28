import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":["Noto Sans CJK SC","DejaVu Sans"],"font.size":9,"axes.unicode_minus":False})
R=json.load(open("R3.json")); S=R["sub"]
NAV="#1F4E79"; GR="#555"
labs={"anticoag":("抗栓方案",{1:"含抗凝药物",0:"仅抗血小板"}),"high_thromb":("高血栓风险",{1:"是",0:"否"}),
      "sep":("Nakaguchi分型",{1:"分隔/小梁型",0:"均质/分层型"}),"age75":("年龄",{1:"≥75岁",0:"<75岁"})}
rows=[("全部患者",None,R["main"]["pt"]["reop_a"],R["main"]["pt"]["reop_b"],R["main"]["pt"]["reop_rd180"],R["main"]["ci"]["reop_rd180"],"812","",True)]
for c,(h,lv) in labs.items():
    it=S[c]["inter_rd"]
    rows.append((h,None,None,None,None,None,"",f"{it[0]*100:+.1f}（{it[1]*100:.1f}～{it[2]*100:.1f}）；P＝{it[3]:.3f}",True))
    for l in (1,0):
        d=S[c][str(l)]; rows.append(("　"+lv[l],None,d["pt"]["reop_a"],d["pt"]["reop_b"],d["pt"]["reop_rd180"],d["ci"]["reop_rd180"],f"{d['n']}（{d['nA']}）","",False))
n=len(rows); fig=plt.figure(figsize=(10,0.36*n+1.2)); ax=fig.add_axes([0.40,0.08,0.26,0.84])
ax.set_ylim(n-0.4,-0.9); ax.set_xlim(-45,25)
ax.axvline(0,color=GR,lw=0.8); ax.axvline(R["main"]["pt"]["reop_rd180"]*100,color=NAV,lw=0.7,ls=":")
for i,(lab,_,a,b,rd,ci,nn,inter,bold) in enumerate(rows):
    fig.text(0.01,0,"")
    ax.text(-128,i,lab,ha="left",va="center",fontweight="bold" if bold and rd is None else "normal",clip_on=False)
    ax.text(-72,i,nn,ha="center",va="center",clip_on=False)
    if rd is not None:
        lo,hi=ci; ax.plot([lo*100,hi*100],[i,i],color=NAV,lw=1.3); ax.plot(rd*100,i,"D" if i==0 else "s",color=NAV,ms=6 if i==0 else 5)
        ax.text(31,i,f"{a*100:.1f} / {b*100:.1f}",ha="left",va="center",clip_on=False)
        ax.text(70,i,f"{rd*100:.1f}（{lo*100:.1f}～{hi*100:.1f}）",ha="left",va="center",clip_on=False)
    if inter: ax.text(70,i,"交互：RD之差 "+inter,ha="left",va="center",color="#8A3B12",fontsize=8,clip_on=False)
for x,t in [(-128,"亚组"),(-72,"例数（MMAE）"),(31,"A / B（%）"),(70,"RD，百分点（95%CI）")]:
    ax.text(x,-0.9,t,ha="left" if x!=-72 else "center",va="center",fontweight="bold",clip_on=False)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.set_yticks([]); ax.set_xlabel("180 d风险差（百分点）")
ax.text(-22,n+1.9,"← 有利于MMAE",ha="center",fontsize=8,color=GR,clip_on=False); ax.text(12,n+1.9,"有利于不行MMAE →",ha="center",fontsize=8,color=GR,clip_on=False)
fig.savefig("f5_forest.png",dpi=300,bbox_inches="tight")
