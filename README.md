[README.txt](https://github.com/user-attachments/files/34058322/README.txt)
# Simulation-Study-of-Target-Trials-Based-on-Cloning-Censoring-and-Weighting
In patients with chronic subdural hematoma (CSDH) who are taking antithrombotic agents, the use of middle meningeal artery embolization (MMAE) as an adjunct within 7 days of burr-hole drainage is associated with a lower risk of reoperation at 180 days.
教学材料 v3（时间有序逐日IPCW＋熵平衡校准版）——全部数据为计算机模拟生成，仅供教学

运行顺序（Python 3.13.17；NumPy 2.5.3、pandas 2.3.3、SciPy 1.18.1、scikit-learn 1.9.1、lifelines 0.30.3、matplotlib 3.11.2）
1. python3 sim2.py        生成基础模拟队列 cohort2.csv（随机种子23）
2. python3 ext3.py        追加CT/MMAE时间戳、计划CT随访、结局可评估性与判定一致性 → cohort3.csv
3. python3 runall3.py     主分析（10插补，500次Bootstrap）、敏感性、亚组、联合策略 → R3.json（约40分钟）
4. python3 diag3.py       权重诊断、重叠、中心×年份、CT随访 → diag3.json
5. python3 riskset.py     逐日风险集与人工删失表 → riskset.json
6. python3 t3.py          表2及补充表S4
7. python3 naive.py       传统分析对照（按实际治疗/第8天标志点）→ naive.json
8. python3 figs2.py; python3 forest.py; python3 figS.py    图1～3及补充图S1～S2
9. python3 ms/build.py    将以上结果填入稿件模板（ms/main.md、ms/supp.md、ms/resp.md）

核心代码：ccw3.py
- hazard_model()：逐日池化Logistic，CT变量自第2天起进入
- weights()：策略A权重1/[1－∏(1－h)]，策略B权重1/∏(1－h)，第8天熵平衡校准
- 其余加权Aalen-Johansen估计沿用ccw.py中的cif_weighted()
旧版脚本（ccw.py、full.py、runall.py等）保留，供对比第2版与第3版的差异
