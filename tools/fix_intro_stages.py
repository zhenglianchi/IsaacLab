# -*- coding: utf-8 -*-
"""Align the three-stage requirements in 3.1 (and one sentence in 3.2.5) with the
chapter's own transient analysis (3.2.2) and the latch mechanism (3.3.5).

Stage requirements, corrected:
  approach  -> compliant (suppress the transient from unshaped inertia)
  contact   -> lateral constraint + damping, while keeping axial insertion capability
  seated    -> actively withdraw (no positional reaction force)
"""
import pathlib
import re

DOC = pathlib.Path('第三章.md')
s = DOC.read_text(encoding='utf-8')

pairs = [
    # (old, new)
    ('因此，同一次装配中，前段要求“刚”，后段要求“柔”，而任何一组单一的固定增益只能在这一连续变化的折中上取某一个点。',
     '就位完成之后，任务对控制器的要求又变为“退出”：此时位置反力已无必要，继续施加只会把已建立的配合重新推入受力状态。'
     '因此同一次装配对阻抗特性的要求是分段相反且互相冲突的：接近段要求柔顺以抑制暂态，接触段要求横向受约束以保证稳定、'
     '同时保留轴向插入能力，就位之后要求主动退出；任何一组单一的固定增益都只能在这几段相反的诉求上取某一个折中点。'),

    ('固定阻抗控制把刚度与阻尼取为常数，只能在自由空间的跟踪能力与接触段的稳定性之间折中，无法随接触状态改变回路特性；',
     '固定阻抗控制把刚度与阻尼取为常数，无法在接近、接触与就位三段之间改变回路特性，'
     '只能在接近段暂态的抑制与接触段的稳定之间折中；'),

    ('在接触事件被可靠识别的前提下，由策略在线重整定任务空间的刚度与阻尼，使控制器在自由段、接触段与就位段各自具备所需的回路特性。',
     '在接触事件被可靠识别的前提下，由策略在线重整定任务空间的刚度与阻尼，使控制器在接近段保持柔顺以抑制暂态、'
     '在接触段具备横向约束与轴向插入能力、并在就位之后主动退出。'),

    ('自由段需要较高刚度，以便在长行程中纠正被式 (3-5) 放大的姿态偏差；',
     '接近段需要柔顺，以抑制由未整形惯量引起的接近段暂态与末端甩动，同时避免在长行程末段带着过大的速度与惯量进入接触；'),

    ('接触段需要较低的转动刚度与足够的阻尼，以便吸收被放大的销尖偏差、避开式 (3-7) 所描述的卡阻分支，并抑制“控制器刚度—接触刚度”回路的极限环；'
     '就位段又需要足够的轴向刚度与力，以克服摩擦与卡阻完成最后一段插入。',
     '接触段需要足够的横向约束与阻尼，以便在吸收被放大的销尖偏差的同时避开式 (3-7) 所描述的卡阻分支、'
     '抑制“控制器刚度—接触刚度”回路的极限环，并在轴向上保留足够的插入能力以克服摩擦与卡阻；'
     '就位完成之后则要求控制器主动退出，不再对已建立的配合施加位置反力。'),
]

done = 0
for old, new in pairs:
    if old in s:
        s = s.replace(old, new, 1)
        done += 1
    else:
        print('  !! 锚点未找到:', old[:34])

DOC.write_text(s, encoding='utf-8')
print('已修正 %d/%d 处' % (done, len(pairs)))

lines = s.splitlines()
i0 = next(i for i, l in enumerate(lines) if l.startswith('## 3.1'))
i1 = next(i for i, l in enumerate(lines) if i > i0 and l.startswith('## 3.2'))
body = '\n'.join(lines[i0:i1])
print('引言无带单位数据 :', '是 ✓' if not re.search(r'\d+(?:\.\d+)?\s*(?:mm|cm|m|N|rad|°|%)', body) else '否 ✗')
print('含“退出”表述    :', '是 ✓' if '退出' in body else '否 ✗')
print('含“刚”字冲突表述:', '否 ✓' if '前段要求“刚”' not in body else '是 ✗')
print('文件行数        :', len(lines))
