#!/usr/bin/env python3
"""NLF accuracy complexity: total inner V-cycles vs desired accuracy eps, for a few graph classes.
The count grows LINEARLY in log(1/eps) -> total work O(m log 1/eps). Data: nlf_accuracy.csv."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL={"web":"#E8743B","social":"#C0392B","road":"#7A8A90","FEM":"#31CBC8"}
MK={"web":"s","social":"D","road":"P","FEM":"^"}
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
rows=list(csv.DictReader(open(D+"nlf_accuracy.csv")))
fig,ax=plt.subplots(figsize=(6.0,4.4))
slopes={}
for cls in ["web","social","road","FEM"]:
    r=[x for x in rows if x["class"]==cls]
    le=np.array([ -np.log10(float(x["eps"])) for x in r ])   # log10(1/eps)
    cy=np.array([ float(x["total_cycles"]) for x in r ])
    sl,ic=np.polyfit(le,cy,1); slopes[cls]=sl
    ax.plot(le,cy,MK[cls],color=COL[cls],ms=8,mec="white",label=f"{cls} (slope {sl:.2f})")
    xx=np.array([le.min(),le.max()]); ax.plot(xx,ic+sl*xx,color=COL[cls],lw=1.2,alpha=0.7)
ax.set_xlabel(r"requested accuracy  $\log_{10}(1/\varepsilon)$")
ax.set_ylabel("total inner V-cycles")
ax.set_title(r"NLF cost $\propto \log(1/\varepsilon)$: total work is $O(m\log 1/\varepsilon)$",fontsize=10.5)
ax.grid(alpha=0.25); ax.legend(fontsize=8.5,loc="upper left")
plt.tight_layout()
out=D+"nlf_accuracy.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
print("wrote",out,"| slopes (cycles per decade of accuracy):",{k:round(v,2) for k,v in slopes.items()})
