#!/usr/bin/env python3
"""NLF max-flow scaling: total wall-clock vs m (setup+solve combined -- the hierarchy is
rebuilt within the algorithm). Point cloud + log-log regression, BK overlaid. Right panel:
continuation steps vs m (flat = O(1), the algorithmic win). Data: nlf_scaling.csv (this
algorithm) + bk_scaling.csv (Boykov-Kolmogorov v3.04, the combinatorial competitor)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

TEAL="#00435A"; AQUA="#31CBC8"; ORANGE="#E8743B"; GREY="#7A8A90"
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
fam=[r for r in csv.DictReader(open(D+"nlf_scaling.csv")) if int(r["ok"])==1]
bk =list(csv.reader(open(D+"bk_scaling.csv")))[1:]
def col(name): return {"grid2d":TEAL,"grid3d":AQUA,"random":ORANGE}.get(name,GREY)

fm=np.array([float(r["m"]) for r in fam]); ft=np.array([float(r["total_s"]) for r in fam])
fs=np.array([int(r["steps"]) for r in fam]); ffam=[r["instance"].split("/")[0] for r in fam]
bm=np.array([float(x[3]) for x in bk]); bt=np.array([float(x[6]) for x in bk])
sf,bf=np.polyfit(np.log(fm),np.log(ft),1); sb,bb=np.polyfit(np.log(bm),np.log(bt),1)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.5))
# --- panel A: total time vs m, log-log ---
# per-edge cost: a flat (slope-0) line in log-x means O(m). Exponent = total-slope - 1.
fpe=ft/fm*1e6; bpe=bt/bm*1e6; spe=sf-1.0; bspe=sb-1.0
for f in ("grid2d","grid3d","random"):
    idx=[i for i in range(len(fm)) if ffam[i]==f]
    ax1.loglog(fm[idx],fpe[idx],"o",color=col(f),ms=7,label=f"NLF {f}")
xx=np.array([fm.min(),fm.max()])
ax1.loglog(xx,np.exp(bf)*xx**sf/xx*1e6,"-",color=TEAL,lw=1.8,alpha=0.7,
           label=f"NLF fit: $t/m\\propto m^{{{spe:.2f}}}$")
ax1.loglog(bm,bpe,"s",color=GREY,ms=6,label="Boykov--Kolmogorov")
xb=np.array([bm.min(),bm.max()])
ax1.loglog(xb,np.exp(bb)*xb**sb/xb*1e6,"--",color=GREY,lw=1.6,label=f"BK fit: $t/m\\propto m^{{{bspe:.2f}}}$")
ax1.set_xlabel("edges  $m$"); ax1.set_ylabel("time per edge  ($\\mu$s/edge)")
ax1.set_title("(A) Per-edge cost flat $\\Rightarrow$ near-$\\mathcal{O}(m)$",fontsize=10.5)
ax1.grid(alpha=0.25,which="both"); ax1.legend(fontsize=7.6,loc="center left",ncol=1)
# --- panel B: continuation steps vs m (flat) ---
for f in ("grid2d","grid3d","random"):
    idx=[i for i in range(len(fm)) if ffam[i]==f]
    ax2.semilogx(fm[idx],fs[idx],"o",color=col(f),ms=7,label=f)
ax2.axhline(np.mean(fs),ls=":",color=GREY,lw=1.4)
ax2.text(fm.max(),np.mean(fs)+0.4,f"mean $={np.mean(fs):.1f}$",ha="right",fontsize=9,color=GREY)
ax2.set_xlabel("edges  $m$"); ax2.set_ylabel("continuation steps to $F^*$")
ax2.set_title("(B) $\\mathcal{O}(1)$ continuation steps (flat in size)",fontsize=10.5)
ax2.set_ylim(0,14); ax2.grid(alpha=0.25,which="both"); ax2.legend(fontsize=8.5)
plt.tight_layout()
out=D+"nlf_scaling.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
print("wrote",out, f"| NLF m^{sf:.3f}  BK m^{sb:.3f}")
