#!/usr/bin/env python3
"""NLF congestion (BPR) scaling: REAL road networks + SYNTHETIC poorly-separable graphs, vs Ipopt.
(A) per-edge time: NLF flat (O(m)) on both families; Ipopt flat on planar roads but rising on
random graphs (sparse-direct KKT fill-in). (B) total wall-clock: Ipopt wins on roads (good
separators), NLF wins + Ipopt blows up on random. Data: nlf_bpr_scaling.csv."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TEAL="#00435A"; AQUA="#31CBC8"; ORANGE="#E8743B"
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
rows=list(csv.DictReader(open(D+"nlf_bpr_scaling.csv")))
def f(x):
    try: return float(x)
    except: return float("nan")
fam=[r["family"] for r in rows]
m=np.array([f(r["m"]) for r in rows]); fa=np.array([f(r["nlf_s"]) for r in rows])
ip=np.array([f(r["ipopt_s"]) for r in rows]); st=[r["ipopt_status"] for r in rows]
ipok=np.array([("SOLVED" in s or "OPTIMAL" in s) for s in st])
# drop the first Ipopt call (JIT/compile warmup): on planar roads Ipopt is ~3-5 us/edge, so a
# road point above 50 us/edge is the one-time compile artifact, not the solver's steady cost.
ipok = ipok & ~(np.array([f=="road" for f in fam]) & (ip/m*1e6 > 50))
def sl(x,y):
    g=np.isfinite(x)&np.isfinite(y)&(y>0); return np.polyfit(np.log(x[g]),np.log(y[g]),1)[0] if g.sum()>=2 else np.nan
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.5))
MK={"road":"o","random":"s"}; LB={"road":"road (real)","random":"random"}
# --- (A) per-edge time ---
for fm in ("road","random"):
    idx=[i for i in range(len(m)) if fam[i]==fm]
    ax1.loglog(m[idx],fa[idx]/m[idx]*1e6,MK[fm],color=TEAL,ms=7,mec="white",label=f"NLF {LB[fm]}")
    ipi=[i for i in idx if ipok[i]]
    ax1.loglog(m[ipi],ip[ipi]/m[ipi]*1e6,MK[fm],color=ORANGE,ms=7,mec="white",label=f"Ipopt {LB[fm]}")
rnd=np.array([i for i in range(len(m)) if fam[i]=="random"]); rip=np.array([i for i in rnd if ipok[i]])
sF=sl(m[rnd],fa[rnd]/m[rnd]); sI=sl(m[rip],ip[rip]/m[rip])
ax1.set_xlabel("edges  $m$"); ax1.set_ylabel("time per edge  ($\\mu$s/edge)")
ax1.set_title(f"(A) Per-edge: NLF flat (O($m$)) vs Ipopt rising on random ($\\propto m^{{{sI:.2f}}}$)",fontsize=9.3)
ax1.grid(alpha=0.25,which="both"); ax1.legend(fontsize=7.5,ncol=2,loc="upper left")
# --- (B) total wall-clock ---
for fm in ("road","random"):
    idx=[i for i in range(len(m)) if fam[i]==fm]
    ax2.loglog(m[idx],fa[idx],MK[fm],color=TEAL,ms=7,mec="white",label=f"NLF {LB[fm]}")
    ipi=[i for i in idx if ipok[i]]
    ax2.loglog(m[ipi],ip[ipi],MK[fm],color=ORANGE,ms=7,mec="white",label=f"Ipopt {LB[fm]}")
fail=[i for i in range(len(m)) if fam[i]=="random" and not ipok[i]]
if fail:
    i=max(fail,key=lambda j:m[j])
    ax2.annotate("Ipopt fails\n(NLF: %.1fs)"%fa[i], xy=(m[i],fa[i]), xytext=(m[i]*0.3,fa[i]*5),
                 fontsize=8,color=TEAL,ha="center",arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.1))
ax2.set_xlabel("edges  $m$"); ax2.set_ylabel("total wall-clock  (s)")
ax2.set_title("(B) Ipopt wins on planar roads; NLF wins + Ipopt blows up on random",fontsize=9.3)
ax2.grid(alpha=0.25,which="both"); ax2.legend(fontsize=7.5,ncol=2,loc="upper left")
plt.tight_layout()
out=D+"nlf_bpr.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
print(f"wrote {out} | NLF random per-edge m^{sF:.2f}  Ipopt random per-edge m^{sI:.2f}")
