import csv, io, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

LOGS = pathlib.Path("logs")
OUT = pathlib.Path("outputs/figs_ch3_new")
OUT.mkdir(parents=True, exist_ok=True)

SRC = {
    "本文方法": "metrics_reeval_ours.csv",
    "C0 固定阻抗": "metrics_reeval_c0.csv",
    "B1 只学阻尼": "metrics_b1_rl_varimp_last.csv",
    "B2 有界残差": "metrics_b2_residual_last.csv",
    "B4 两阶段": "metrics_b4_twostage_last.csv",
}
DATA = {}
for name, f in SRC.items():
    p = LOGS / f
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8-sig")))
    suc = [r for r in rows if r.get("success") == "1"]
    F = sorted(float(r["wrist_F_max_N"]) for r in suc)
    q = lambda v, pp: float(np.percentile(v, pp * 100)) if v else float("nan")
    att = [float(r["attitude_rms_rad"]) for r in suc if r.get("attitude_rms_rad") not in (None, "", "nan")]
    mx = lambda c: max(float(r[c]) for r in suc if r.get(c) not in (None, "", "nan"))
    DATA[name] = dict(
        rate=100.0 * len(suc) / len(rows), n=len(suc),
        med=q(F, 0.5), p90=q(F, 0.9), mx=max(F), F=F,
        att_med=float(np.median(att)) if att else float("nan"),
        M=max(mx("wrist_Mx_max_Nm"), mx("wrist_My_max_Nm"), mx("wrist_Mz_max_Nm")),
    )

names = list(SRC.keys())
COL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
DPI = 200

# ---- 图3-2（1120×704 像素）----
fig = plt.figure(figsize=(1120 / DPI, 704 / DPI), dpi=DPI)
ax1 = fig.add_axes([0.07, 0.16, 0.38, 0.76])
ax2 = fig.add_axes([0.55, 0.16, 0.42, 0.76])
x = np.arange(len(names))
ax1.bar(x, [DATA[n]["rate"] for n in names], color=COL, width=0.62)
for i, n in enumerate(names):
    ax1.text(i, DATA[n]["rate"] + 1.5, "%.1f" % DATA[n]["rate"], ha="center", fontsize=9)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8.5, rotation=18)
ax1.set_ylabel("装配成功率 / %", fontsize=10); ax1.set_ylim(0, 112)
ax1.grid(axis="y", alpha=0.3); ax1.set_axisbelow(True)
w = 0.26
ax2.bar(x - w, [DATA[n]["med"] for n in names], width=w, label="中位数", color="#4c72b0")
ax2.bar(x, [DATA[n]["p90"] for n in names], width=w, label="90% 分位", color="#dd8452")
ax2.bar(x + w, [DATA[n]["mx"] for n in names], width=w, label="最大值", color="#c44e52")
ax2.axhline(40, ls="--", lw=1.2, color="k")
ax2.text(len(names) - 0.5, 41.5, "40 N 判据线", fontsize=8.5, ha="right")
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8.5, rotation=18)
ax2.set_ylabel("腕部合力峰值 / N", fontsize=10); ax2.legend(fontsize=8.5, frameon=False)
ax2.grid(axis="y", alpha=0.3); ax2.set_axisbelow(True)
fig.savefig(OUT / "fig3-2.png", dpi=DPI); plt.close(fig)

# ---- 图3-3（1200×704）----
fig = plt.figure(figsize=(1200 / DPI, 704 / DPI), dpi=DPI)
ax = fig.add_axes([0.09, 0.16, 0.87, 0.78])
for i, n in enumerate(names):
    F = DATA[n]["F"]
    y = np.arange(1, len(F) + 1) / len(F)
    ax.step(F, y, where="post", color=COL[i], lw=1.8, label="%s（最坏 %.1f N）" % (n, DATA[n]["mx"]))
ax.axvline(40, ls="--", lw=1.3, color="k")
ax.text(40.8, 0.06, "40 N 判据线", fontsize=9)
ax.set_xlabel("成功回合的最坏腕部合力峰值 / N", fontsize=10)
ax.set_ylabel("累积分布", fontsize=10); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=8.2, frameon=False, loc="lower right")
fig.savefig(OUT / "fig3-3.png", dpi=DPI); plt.close(fig)

# ---- 图3-4（1920×704）----
fig = plt.figure(figsize=(1920 / DPI, 704 / DPI), dpi=DPI)
axes = [fig.add_axes([0.06 + k * 0.315, 0.19, 0.26, 0.73]) for k in range(3)]
axes[0].bar(x, [DATA[n]["mx"] for n in names], color=COL, width=0.6)
axes[0].axhline(40, ls="--", lw=1.1, color="k")
axes[0].set_title("最坏腕部合力峰值 / N", fontsize=10)
axes[1].bar(x, [DATA[n]["att_med"] for n in names], color=COL, width=0.6)
axes[1].set_title("姿态残差均方根中位数 / rad", fontsize=10)
axes[2].bar(x, [DATA[n]["M"] for n in names], color=COL, width=0.6)
axes[2].set_title("腕部力矩峰值最大 / N·m", fontsize=10)
for a in axes:
    a.set_xticks(x); a.set_xticklabels(names, fontsize=7.6, rotation=20)
    a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)
fig.savefig(OUT / "fig3-4.png", dpi=DPI); plt.close(fig)

for f in ("fig3-2.png", "fig3-3.png", "fig3-4.png"):
    im = Image.open(OUT / f)
    print("  生成 %s  %dx%d" % (f, im.size[0], im.size[1]))

# ---- 替换文档中的图片（保持原显示尺寸）----
import shutil
from docx import Document

DOC = pathlib.Path(r"C:\Users\zhenglianchi\Desktop\IsaacLab\第三章.docx")
shutil.copy2(DOC, str(DOC).replace(".docx", "_备份_before_figs.docx"))
d = Document(str(DOC))
for rid, f in (("rId270", "fig3-2.png"), ("rId271", "fig3-3.png"), ("rId272", "fig3-4.png")):
    part = d.part.related_parts[rid]
    part._blob = (OUT / f).read_bytes()
    print("  已替换 %s <- %s" % (rid, f))
d.save(str(DOC))
print("  文档已保存 ✓")

d2 = Document(str(DOC))
for sh in d2.element.body.iter():
    if sh.tag.endswith("}inline"):
        ext = sh.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
        blip = sh.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
        rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
        im = Image.open(io.BytesIO(d2.part.related_parts[rid].blob))
        print("  核对 %s: 像素 %dx%d 显示 %.1f×%.1f cm" % (rid, im.size[0], im.size[1],
              int(ext.get("cx")) / 360000, int(ext.get("cy")) / 360000))
