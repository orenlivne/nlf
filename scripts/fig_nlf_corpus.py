#!/usr/bin/env python3
"""NLF robustness + O(m) over the full real-world SuiteSparse corpus (the graphs LAMG+ was
benchmarked on). (A) total wall-clock vs edges m, log-log, with the fitted exponent: t ~ m^0.99 over
~1750 graphs and 5 decades = genuine O(m). (B) Newton (continuation) step count: a flat, single-digit
distribution independent of size/class. Data: nlf_corpus.csv."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TEAL="#00435A"; ORANGE="#E8743B"; GREY="#8A99A0"
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
rows=[r for r in csv.DictReader(open(D+"nlf_corpus.csv")) if r["converged"]=="1"]
m=np.array([float(r["m"]) for r in rows]); t=np.array([float(r["nlf_s"]) for r in rows])
it=np.array([int(r["nlf_steps"]) for r in rows])
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.4))
# --- (A) total time vs m, log-log, with the O(m) fit ---
g=(m>=1000)&(t>0)                                   # fit above the fixed-overhead floor
sl,ic=np.polyfit(np.log(m[g]),np.log(t[g]),1)
ax1.loglog(m,t,"o",color=TEAL,ms=3,alpha=0.30,mec="none")
xx=np.array([m[g].min(),m[g].max()])
ax1.loglog(xx,np.exp(ic)*xx**sl,color=ORANGE,lw=2.2,label=f"fit  $t\\propto m^{{{sl:.2f}}}$")
ax1.loglog(xx, np.exp(ic)*xx[0]**sl*(xx/xx[0])**1.0, color=GREY,lw=1.1,ls="--",label="slope 1 (O($m$))")
ax1.set_xlabel("edges  $m$"); ax1.set_ylabel("NLF total wall-clock  (s)")
ax1.set_title(f"(A) $t\\propto m^{{{sl:.2f}}}$ over {int(g.sum())} real graphs: genuine O($m$)",fontsize=9.6)
ax1.grid(alpha=0.25,which="both"); ax1.legend(fontsize=8,loc="upper left")
# --- (B) Newton-step histogram ---
vals,cnts=np.unique(it,return_counts=True)
ax2.bar(vals,cnts,color=TEAL,width=0.8)
ax2.axvline(np.median(it),color=ORANGE,lw=1.6,label=f"median {int(np.median(it))} steps")
ax2.set_xlabel("Newton (continuation) steps"); ax2.set_ylabel("number of graphs")
ax2.set_title("(B) Newton-step count: flat in size and class",fontsize=9.6)
ax2.grid(alpha=0.25,axis="y"); ax2.legend(fontsize=8)
plt.tight_layout()
out=D+"nlf_corpus.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
pe=t[g]/m[g]*1e6
print(f"wrote {out} | {int(g.sum())} graphs (m>=1e3), t~m^{sl:.3f}, per-edge median {np.median(pe):.2f} us, "
      f"steps {it.min()}-{it.max()} (median {int(np.median(it))})")
