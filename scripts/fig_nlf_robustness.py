#!/usr/bin/env python3
"""NLF robustness across graph classes (LAMG+ test set). Per-edge wall-clock vs edges m, colored by
graph class: flat (O(m)) and bounded across structured/FEM/web/social/road/citation -- NLF inherits
LAMG+'s graph-class robustness. Data: nlf_robustness.csv."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
rows=list(csv.DictReader(open(D+"nlf_robustness.csv")))
def f(x):
    try: return float(x)
    except: return float("nan")
CLS={"structured":("#00435A","o"),"FEM":("#31CBC8","^"),"FEM/aniso":("#31CBC8","v"),
     "web":("#E8743B","s"),"social":("#C0392B","D"),"road":("#7A8A90","P"),"citation":("#6C3483","*")}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.4))
seen=set()
for r in rows:
    cls=r["class"]; m=f(r["m"]); pe=f(r["per_edge_us"]); it=f(r["nlf_steps"])
    col,mk=CLS.get(cls,("#333","o")); lab=cls if cls not in seen else None; seen.add(cls)
    ax1.semilogx([m],[pe],mk,color=col,ms=9,mec="white",label=lab)
    ax2.semilogx([m],[it],mk,color=col,ms=9,mec="white",label=lab)
ax1.set_xlabel("edges  $m$"); ax1.set_ylabel("NLF time per edge  ($\\mu$s/edge)")
ax1.set_title("(A) Per-edge cost: flat (O($m$)) across graph classes",fontsize=10)
ax1.set_ylim(0,None); ax1.grid(alpha=0.25,which="both"); ax1.legend(fontsize=7.5,ncol=2)
ax2.set_xlabel("edges  $m$"); ax2.set_ylabel("Newton (continuation) steps")
ax2.set_title("(B) Newton-step count: flat in size and class",fontsize=10)
ax2.set_ylim(0,None); ax2.grid(alpha=0.25,which="both"); ax2.legend(fontsize=7.5,ncol=2)
plt.tight_layout()
out=D+"nlf_robustness.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
mm=np.array([f(r["m"]) for r in rows]); pe=np.array([f(r["per_edge_us"]) for r in rows])
print(f"wrote {out} | {len(rows)} graphs, per-edge range {np.nanmin(pe):.1f}-{np.nanmax(pe):.1f} us/edge, m up to {np.nanmax(mm):.2e}")
