import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({"font.family":["Noto Sans CJK SC","DejaVu Sans"],"font.size":9,"axes.unicode_minus":False})
EC="#33415C"
def box(ax,x,y,w,h,t,fs=8.3,fc="#F4F6F8",ha="center"):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=EC,lw=0.9))
    ax.text(x if ha=="center" else x-w/2+0.12,y,t,ha=ha,va="center",fontsize=fs,linespacing=1.45)
def arr(ax,x1,y1,x2,y2): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.9,color=EC))
# Fig1 flow
fig,ax=plt.subplots(figsize=(7.4,7.2),dpi=300); ax.set_xlim(0,10); ax.set_ylim(0,10.4); ax.axis("off")
box(ax,2.9,9.85,4.9,0.8,"2020年1月至2025年3月两中心行钻孔引流术的\n症状性慢性硬膜下血肿患者（n＝2 317）")
arr(ax,2.9,9.45,2.9,8.95)
box(ax,2.9,8.55,4.9,0.8,"入院前规律服用抗血小板和（或）抗凝药物≥4周\n且有长期抗栓指征（n＝912）")
arr(ax,2.9,8.15,2.9,6.05); ax.plot([2.9,5.55],[7.1,7.1],color=EC,lw=0.9); arr(ax,5.55,7.1,5.72,7.1)
box(ax,7.85,7.1,4.2,1.9,"排除（n＝100）：\n继发性CSDH（开颅、分流术后等）41例\n合并颅内肿瘤或凝血功能障碍 23例\n行开颅或神经内镜手术 21例\n术前已行MMAE 5例\n手术日关键资料缺失 10例",fs=7.8,fc="#FFFFFF")
box(ax,2.9,5.65,4.9,0.8,"纳入分析（n＝812）；时间零点：手术日\n每例患者复制为两个克隆，分别分配至两种策略",fc="#E9EEF5")
arr(ax,1.5,5.25,1.5,4.6); arr(ax,4.45,5.25,4.45,4.6)
box(ax,1.5,4.05,2.8,1.0,"策略A：术后7 d内行MMAE\n（克隆n＝812）",fc="#E3ECF7")
box(ax,4.45,4.05,2.8,1.0,"策略B：术后7 d内不行MMAE\n（克隆n＝812）",fc="#F2F2F2")
arr(ax,1.5,3.55,1.5,3.02); arr(ax,4.45,3.55,4.45,3.02)
box(ax,1.5,2.15,2.8,1.7,"7 d内行MMAE 275例\n宽限期内结局3例计入\n（再手术2例，死亡1例）\n第7天末未栓塞且无事件\n→人工删失535例\n第8天起加权随访274例",fs=7.2)
box(ax,4.45,2.15,2.8,1.7,"7 d内行MMAE者于栓塞前\n一日末人工删失275例\n宽限期内结局2例计入\n（均为再手术）\n第8天起加权随访535例\n（其中＞7 d行MMAE 12例）",fs=7.2)
arr(ax,1.5,1.28,1.5,0.92); arr(ax,4.45,1.28,4.45,0.92)
box(ax,2.97,0.5,5.9,0.75,"加权后随访至术后180 d\n失访删失30例（3.7%；7 d内行MMAE者9例，未行者21例）",fs=7.8)
fig.savefig("f1_flow.png",bbox_inches="tight"); plt.close()
# Fig2 CCW schematic
fig,ax=plt.subplots(figsize=(7.6,3.4),dpi=300); ax.set_xlim(-75,195); ax.set_ylim(-0.9,5.4); ax.axis("off")
ax.add_patch(Rectangle((0,0.3),7,4.6,fc="#FFF4E0",ec="none")); ax.text(3.5,-0.5,"宽限期 0～7 d",ha="center",va="top",fontsize=7,color="#8A5A00")
ax.plot([0,0],[0.3,4.9],color=EC,lw=1.2); ax.plot([0,182],[0.3,0.3],color="#888",lw=0.8)
ax.text(-2,-0.1,"0（手术日）",ha="right",va="top",fontsize=7.5)
for x in (30,90,180): ax.plot([x,x],[0.25,0.35],color="#888"); ax.text(x,-0.1,f"{x} d",ha="center",va="top",fontsize=7.5)
CA="#1F4E79"; CB="#666666"; RED="#B03A2E"
rows=[(4.3,"策略A克隆：第3天行MMAE",CA,180,None,3),(3.3,"策略A克隆：7 d内未行MMAE",CA,7,"人工删失（第7天末仍未栓塞）",None),
      (2.1,"策略B克隆：第3天行MMAE",CB,2,"人工删失（栓塞前一日末）",3),(1.1,"策略B克隆：7 d内未行MMAE",CB,180,None,None)]
for y,lab,c,end,cz,m in rows:
    ax.text(-4,y,lab,ha="right",va="center",fontsize=7.8,color=c)
    ax.plot([0,end],[y,y],color=c,lw=2.6,solid_capstyle="butt")
    if m: ax.plot(m,y+0.22,marker="v",color=c,ms=5); ax.text(m+1.5,y+0.32,"MMAE",fontsize=7,color=c)
    if cz: ax.plot(end,y,marker="x",color=RED,ms=7,mew=1.6); ax.text(end+4,y,cz,va="center",fontsize=7.5,color=RED)
ax.text(181,4.3+0.3,"第8天起已栓塞克隆权重＝1/P(7 d内栓塞)",ha="right",fontsize=7,color=CA)
ax.text(181,1.1+0.3,"未删失克隆权重＝1/P(截至当日仍未栓塞)",ha="right",fontsize=7,color=CB)
ax.text(-74,5.05,"同一患者在时间零点被复制为两个克隆",fontsize=7.5,color=EC)
fig.savefig("f2_ccw.png",bbox_inches="tight"); plt.close()
# Fig3 Love
b=json.load(open("bal.json")); items=sorted(b.items(),key=lambda kv:abs(kv[1][0]))
fig,ax=plt.subplots(figsize=(5.2,7.4),dpi=300); y=np.arange(len(items))
ax.scatter([abs(v[0]) for k,v in items],y,facecolors="white",edgecolors="#B03A2E",s=22,label="加权前（按实际接受治疗）",zorder=3)
ax.scatter([abs(v[1]) for k,v in items],y,marker="s",color="#1F4E79",s=18,label="加权后（策略A与策略B克隆）",zorder=3)
for i,(k,v) in enumerate(items): ax.plot([abs(v[1]),abs(v[0])],[i,i],color="#BBBBBB",lw=0.6,zorder=1)
ax.axvline(0.1,ls="--",color="#555",lw=0.8); ax.set_yticks(y); ax.set_yticklabels([k for k,v in items],fontsize=8)
ax.set_xlabel("标准化均数差绝对值（|SMD|）"); ax.set_xlim(0,0.7); ax.legend(frameon=False,loc="lower right",fontsize=7.5)
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.savefig("f3_love.png",bbox_inches="tight"); plt.close()
# Fig4 CIF with bands
R=json.load(open("R3.json"))["main"]; pt=R["pt"]
cA=np.array(pt["curveA"])*100; cB=np.array(pt["curveB"])*100; t=np.arange(181)
fig,ax=plt.subplots(figsize=(6.2,4.2),dpi=300)
ax.fill_between(t,np.array(R["bandB"][0])*100,np.array(R["bandB"][1])*100,step="post",color="#7F7F7F",alpha=0.15,lw=0)
ax.fill_between(t,np.array(R["bandA"][0])*100,np.array(R["bandA"][1])*100,step="post",color="#1F4E79",alpha=0.15,lw=0)
ax.step(t,cB,where="post",color="#555555",lw=1.7,label="策略B：术后7 d内不行MMAE")
ax.step(t,cA,where="post",color="#1F4E79",lw=1.7,label="策略A：术后7 d内行MMAE")
ax.axvspan(0,7,color="#FFF4E0",zorder=0); ax.text(3.5,25,"宽限期",ha="center",va="center",rotation=90,fontsize=7.5,color="#8A5A00")
ax.set_xlim(0,180); ax.set_ylim(0,30); ax.set_xticks(range(0,181,30)); ax.set_yticks(range(0,31,5))
ax.set_xlabel("术后天数（d）"); ax.set_ylabel("症状性复发再手术累积发生率（%）"); ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0.07,1.0),fontsize=8)
ci=R["ci"]
ax.text(178,2.0,f"180 d：{pt['reop_a']*100:.1f}% vs {pt['reop_b']*100:.1f}%\n风险差 {pt['reop_rd180']*100:.1f}个百分点（95%CI：{ci['reop_rd180'][0]*100:.1f}～{ci['reop_rd180'][1]*100:.1f}）\n风险比 {pt['reop_rr']:.2f}（95%CI：{ci['reop_rr'][0]:.2f}～{ci['reop_rr'][1]:.2f}）",ha="right",fontsize=7.8)
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.savefig("f4_cif.png",bbox_inches="tight"); plt.close()
print("ok")
