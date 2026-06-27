#!/usr/bin/env python3
"""NLF congestion-flow scaling vs Ipopt (interior-point, sparse-direct core).
(A) per-edge time: NLF flat (O(m)) vs Ipopt rising (superlinear) on poorly-separable graphs.
(B) total wall-clock crossover, Ipopt timeouts marked. Data: nlf_traffic.csv."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TEAL="#00435A"; AQUA="#31CBC8"; ORANGE="#E8743B"; GREY="#7A8A90"
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
rows=list(csv.DictReader(open(D+"nlf_traffic.csv")))
def fam(r): return r["instance"].split("/")[0]
def f(x):
    try: return float(x)
    except: return float("nan")
m=np.array([f(r["m"]) for r in rows]); fa=np.array([f(r["nlf_s"]) for r in rows])
ip=np.array([f(r["ipopt_s"]) for r in rows]); st=[r["ipopt_status"] for r in rows]
fams=[fam(r) for r in rows]; ipok=np.array([("SOLVED" in s or "OPTIMAL" in s) for s in st])
def sl(x,y):
    g=np.isfinite(x)&np.isfinite(y)&(y>0); return np.polyfit(np.log(x[g]),np.log(y[g]),1)[0] if g.sum()>=2 else np.nan
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.5))
mk={"grid2d":"o","random":"s","grid3d":"^"}
# --- (A) per-edge time, log-log ---
for fm in ("grid2d","random"):
    idx=[i for i in range(len(m)) if fams[i]==fm]
    ax1.loglog(m[idx],fa[idx]/m[idx]*1e6,mk[fm],color=TEAL,ms=7,mec="white",label=f"NLF {fm}")
    ipi=[i for i in idx if ipok[i]]
    ax1.loglog(m[ipi],ip[ipi]/m[ipi]*1e6,mk[fm],color=ORANGE,ms=7,mec="white",label=f"Ipopt {fm}")
# fit slopes on random (poorly-separable)
ridx=np.array([i for i in range(len(m)) if fams[i]=="random"])
ripi=np.array([i for i in ridx if ipok[i]])
sF=sl(m[ridx],fa[ridx]/m[ridx]); sI=sl(m[ripi],ip[ripi]/m[ripi])
ax1.set_xlabel("edges  $m$"); ax1.set_ylabel("time per edge  ($\\mu$s/edge)")
ax1.set_title(f"(A) Per-edge: NLF flat ($\\propto m^{{{sF:.2f}}}$, O($m$)) vs Ipopt rising ($\\propto m^{{{sI:.2f}}}$)",fontsize=9.3)
ax1.grid(alpha=0.25,which="both"); ax1.legend(fontsize=7.5,ncol=2,loc="upper left")
# --- (B) total time crossover, timeouts marked ---
for fm in ("grid2d","random"):
    idx=[i for i in range(len(m)) if fams[i]==fm]
    ax2.loglog(m[idx],fa[idx],mk[fm],color=TEAL,ms=7,mec="white",label=f"NLF {fm}")
    ipi=[i for i in idx if ipok[i]]; ito=[i for i in idx if not ipok[i] and np.isfinite(ip[i])]
    ax2.loglog(m[ipi],ip[ipi],mk[fm],color=ORANGE,ms=7,mec="white",label=f"Ipopt {fm}")
    if ito: ax2.loglog(m[ito],ip[ito],"x",color="red",ms=10,mew=2,label="Ipopt timeout/OOM" if fm=="random" else None)
ax2.set_xlabel("edges  $m$"); ax2.set_ylabel("total wall-clock  (s)")
ax2.set_title("(B) NLF wins on poorly-separable graphs, widening",fontsize=10)
# mark the largest random instance where Ipopt failed (TIME_LIMIT) but NLF solved
fail=[i for i in range(len(m)) if fams[i]=="random" and not ipok[i]]
if fail:
    i=max(fail,key=lambda j:m[j])
    ax2.annotate("Ipopt fails\n(NLF: %.1fs)"%fa[i], xy=(m[i],fa[i]), xytext=(m[i]*0.32,fa[i]*4.5),
                 fontsize=8,color=TEAL,ha="center",arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.1))
ax2.grid(alpha=0.25,which="both"); ax2.legend(fontsize=7.5,ncol=1,loc="upper left")
plt.tight_layout()
out=D+"nlf_traffic.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
print(f"wrote {out} | NLF per-edge m^{sF:.2f}  Ipopt per-edge m^{sI:.2f} (random)")
