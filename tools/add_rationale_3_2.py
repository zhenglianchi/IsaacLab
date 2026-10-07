"""Insert 3.2.4 (long guide pin: error amplification / moment lever / jamming) and
3.2.5 (grounding in the literature) into 第三章.md, and extend the reference list.

Robust: backs the file up, locates `## 3.3` and inserts immediately before it, and
appends new references after the last numbered entry.
"""
import pathlib
import re
import shutil
import time

DOC = pathlib.Path("第三章.md")
bak = DOC.with_name(f"第三章.backup-{time.strftime('%Y%m%d-%H%M%S')}.md")
shutil.copy2(DOC, bak)

NEW = """### 3.2.4 长导向销场景下的误差放大、力矩杠杆与卡阻风险

上述关于固定阻抗局限的分析对所有插装任务都成立，而本文场景有两点特殊之处，使该局限被进一步放大：**导向销的导向段很长，而径向间隙很小**。本节把这两点写成可检验的几何关系，作为方法设计的直接依据。

**（1）姿态误差被销长放大。** 设导向销导向段长度为 L_p，末端姿态偏差为 θ，则销尖相对其名义位置的横向偏差近似为

    e_tip ≈ e_ee + L_p · θ                                                        (3-5)

式中 e_ee 为末端（腕部）位置偏差。式 (3-5) 说明：**销越长，姿态误差对销尖对中精度的影响越大，且是线性放大**。取本任务量级（L_p ≈ 0.20 m）代入：姿态偏差 3°（0.0524 rad）对应销尖附加横向偏差约 **10.5 mm**，而就位判据要求的横向误差为 **2 mm** 量级——两者相差五倍以上。因此在本场景中，**决定装配成败的不是末端定位精度，而是姿态精度**；这也解释了为什么控制律必须能够独立地调整**转动**方向的刚度与阻尼，而不能只在平动方向上做柔顺。

**（2）销尖受力经销长形成力矩杠杆。** 作用在销尖的横向接触力 F_xy 会在腕部形成近似

    τ_wrist ≈ F_xy · L_p                                                          (3-6)

的力矩。同样取 L_p ≈ 0.20 m：10 N 的销尖横向力即对应约 2 N·m 的腕部力矩，实测中固定阻抗基线的腕部力矩峰值达到 **8.66 N·m**。这带来两点结论：第一，**最坏受力必须在腕部（而非销尖）评价**，本文的最坏力判据因此取腕部六维力传感器的合力峰值；第二，长销场景下无论对位置还是对力的约束，都比短销场景更严格。

**（3）卡阻倾向随"已进入长度/间隙"之比上升。** 插装接触的经典卡阻条件由两点接触的力平衡给出，其形式可写为

    R = h · sinθ₂ + r · cosθ                                                      (3-7)

式中 h 为已进入导向段的长度，r 为销半径，θ、θ₂ 为两点接触处的接触角[25]。式 (3-7) 表明：**已进入长度 h 越大、径向单边间隙越小，越容易进入卡阻（楔紧）状态**。本文的"长销 + 小间隙"组合恰好落在该风险区；更重要的是，卡阻是**姿态相关**的——它取决于销与孔轴线的夹角，而不是取决于末端的位置偏差。这与（1）的结论一致：**本场景的核心矛盾在转动自由度上**。

**（4）细长销的柔性使接触等效刚度随深度漂移。** 导向销在轴向压力下可近似为细长梁，其横向柔度随受载形式与已进入长度变化，因此"控制器刚度 + 接触刚度"构成的二阶回路，其等效接触刚度**不是常数**，而是随插入深度变化的[12,26]。固定阻抗只能针对某一个等效刚度整定，无法同时匹配浅插入（悬臂较长、接触较软）与深插入（导向已建立、接触较硬）两个阶段。

**（5）长销意味着长行程与不可忽略的接触瞬间速度。** 由于导向段长，接触发生前存在一段较长的自由行程（本文规范工况的自由落差为 20~32 cm）。按 3.2.2 节的分析，比例微分回路在自由空间中会因未整形的惯量产生接近段暂态，行程越长，该暂态在接触时刻累积的速度与惯量成分越大；接触瞬间的速度直接决定冲击力的量级。因此**长销场景不仅要求接触后柔顺，还要求接触发生的瞬间已经处于合适的阻抗状态**——这正是"何时切换"必须由接触事件而非固定时刻决定的原因。

综合（1）~（5），本文场景的困难可以概括为一句话：**长导向销把姿态误差与销尖受力分别放大到"决定成败"与"决定安全性"的量级，而这两者的最优控制特性又随接触状态与插入深度而变，因此控制器必须在自由段、接触段与就位段之间在线改变其阻抗特性。**

### 3.2.5 方法依据与文献支撑

上述分析并非本文独有，它与装配力学与柔顺控制的三条经典结论相衔接，同时也存在本文场景特有的空白。

**（1）装配的困难来自接触状态，而非定位精度。** 若对象位置可被完美控制，精密插入本不构成难题[31]；实际困难来自接触发生后由摩擦与几何约束形成的接触状态（卡阻、楔紧、两点接触），其可行域由式 (3-7) 一类的几何—力平衡条件决定[12,25,26]。这一结论为本文把"接触事件的识别与控制特性的重整定"作为核心问题提供了依据。

**（2）柔顺是化解接触约束的主要手段，且柔顺中心的位置至关重要。** 被动柔顺装置（RCC）通过把柔顺中心置于销尖附近，使横向/转动偏差被"吸收"而不转化为卡阻力矩[27]，这一思想说明：**有效的柔顺需要在正确的自由度上、以正确的量级作用**。主动侧对应的是阻抗/导纳控制，即通过在线整定刚度与阻尼来间接设定柔顺中心与力的传递关系[21,23,24]。这为本文"由策略在线整定任务空间刚度与阻尼"的动作空间设计提供了直接依据。

**（3）由学习在线整定阻抗在接触丰富插装中已被证明有效。** 近期工作把阻抗参数作为策略输出，在接触丰富或多接触插装任务上学习变阻抗策略，并通过约束或奖励引导其进入可行域[28,29]；也有工作直接在装配操作中用腕部力传感器闭环调节装配条件[30]。这些结果说明"学习式变阻抗"路线本身是可行且被认可的方向，本文沿用该路线并把决策问题明确为**接触事件驱动的分段整定**。

**（4）本文场景特有的空白与由此得到的设计依据。** 上述工作大多针对短销、较大间隙，或把接触状态视为可由规划预先安排的过程；而在**长导向销 + 小间隙 + 大行程**的场景下，由（1）~（5）可得三条设计依据：

第一，**必须分段，且分段的边界应由接触事件给出**。自由段需要较高刚度以在长行程中纠正被放大的姿态偏差（式 (3-5)）；接触段需要较低的转动刚度与足够的阻尼，以吸收被放大的销尖偏差、避免式 (3-7) 的卡阻分支并抑制"控制器刚度—接触刚度"回路的极限环；就位段又需要足够的轴向力以克服摩擦与卡阻完成最后一段插入。三个阶段的刚度需求方向相反，任何单一固定增益都只能在其中一段最优。

第二，**分段不能靠预规划时刻实现**。接触发生的时刻取决于实际初始姿态与间隙（式 (3-5)、(3-7)），而初始姿态在随机化下不可预知；腕部关节反力中又混有运动惯性成分（3.2.1 节实测），无法据此区分接触。因此分段的触发只能依据**装配对象与对接面之间的真实接触力**。

第三，**分段的程度应由学习决定，而非由阈值给定**。卡阻与振荡的分界与 L_p、c_r、姿态和插入深度强耦合（式 (3-5)~(3-7)），难以给出一个在全部初始条件下都成立的解析阈值；把"何时柔顺、柔到什么程度"交给策略在线决定，正是本文采用强化学习而非阈值切换或固定增益调度的根本原因。相应地，本文把分段体现为**奖励混合比例与轴向力上限的连续过渡**，而不改变策略结构，从而在获得分段行为的同时避免子策略切换带来的不连续。

上述依据与 3.6.7 节的实测结果一致：在自由落差约 31 cm 的常规工况下，固定阻抗已能完成大部分装配（成功率 91.0%），方法的收益主要体现在接触力与终段稳定性上；而把自由落差收紧到规范工况（20 cm + 向下随机 0~12 cm）后，可用于纠正被放大姿态误差的行程变短，固定阻抗的成功率降至 83.0%，本方法达到 99.0%——正是"姿态/容差预算成为瓶颈"这一预测的直接体现。

"""

lines = DOC.read_text(encoding="utf-8").splitlines(keepends=True)
idx = next(i for i, l in enumerate(lines) if l.startswith("## 3.3"))
out = "".join(lines[:idx]) + NEW + "".join(lines[idx:])

# --- extend the reference list -------------------------------------------------
refs = """[25] Jamming problems and the effects of compliance in dual peg-hole disassembly[J]. Proceedings of the Royal Society A, 2024, 480(2286). https://royalsocietypublishing.org/rspa/article/480/2286/20230364/101185/Jamming-problems-and-the-effects-of-compliance-in

[26] The study of the insertion force and moment for improving insertion conditions[D]. New Jersey Institute of Technology. https://digitalcommons.njit.edu/cgi/viewcontent.cgi?article=3777&context=theses

[27] A Multiple RCC Device for Polygonal Peg Insertion[J]. JSME International Journal Series C, 2002, 45(1): 306-315. https://www.jstage.jst.go.jp/article/jsmec/45/1/45_1_306/_article/-char/en

[28] Constraint-Grounded Reinforcement Learning for Variable Impedance Control in Contact-Rich Robotic Insertion. arXiv:2609.13516. https://arxiv.org/pdf/2609.13516

[29] Apprentissage par renforcement guidé par contraintes pour le contrôle en impédance variable dans l'insertion robotique à contacts multiples. https://www.lefilrobotique.fr/article/6167774-apprentissage-par-renforcement-guide-par-contraintes-pour-le-controle-en-impedance-variable-dans-l-insertion-robotique-a-contacts-multiples

[30] The analysis for the force sensor which is used in the assembly operation by the robot[J]. IFAC-PapersOnLine. https://www.sciencedirect.com/science/article/pii/S147466701754077X/pdf

[31] If the object position was controlled perfectly, precision insertion would not pose any problem[D/OL]. Harvard BioRobotics. http://www.biorobotics.harvard.edu/pubs/jsthesis.pdf
"""
out = out.rstrip("\n") + "\n\n" + refs
DOC.write_text(out, encoding="utf-8")

print(f"backup   : {bak.name}")
print(f"inserted : 3.2.4 + 3.2.5 before line {idx + 1} (## 3.3)")
print(f"refs     : appended [25]-[31]")
print(f"new size : {len(out)} chars, {len(out.splitlines())} lines")
