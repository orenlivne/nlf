#!/usr/bin/env python3
"""Multicommodity NLF (K=4) over the full real-world SuiteSparse corpus, mirroring the
single-commodity corpus figure. (A) total wall-clock vs edges m, log-log, fitted exponent +
slope-1 reference, with the single-commodity sweep (grey) underneath for the K-cost contrast.
(B) chord-step count histogram. Data: nlf_mc_corpus.csv (+ nlf_corpus.csv for the overlay)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TEAL="#00435A"; ORANGE="#E8743B"; GREY="#8A99A0"; AQUA="#31CBC8"
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
rows=[r for r in csv.DictReader(open(D+"nlf_mc_corpus.csv")) if r["converged"]=="1"]
m=np.array([float(r["m"]) for r in rows]); t=np.array([float(r["nlf_s"]) for r in rows])
it=np.array([int(r["nlf_steps"]) for r in rows]); su=np.array([int(r["setups"]) for r in rows])
rows1=[r for r in csv.DictReader(open(D+"nlf_corpus.csv")) if r["converged"]=="1"]
m1=np.array([float(r["m"]) for r in rows1]); t1=np.array([float(r["nlf_s"]) for r in rows1])
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.4))
# --- (A) total time vs m, log-log, with the O(m) fit; single-commodity sweep underneath ---
g=(m>=1000)&(t>0)
sl,ic=np.polyfit(np.log(m[g]),np.log(t[g]),1)
ax1.loglog(m1,t1,"o",color=GREY,ms=2.2,alpha=0.18,mec="none",label="single commodity (§4)")
ax1.loglog(m,t,"o",color=TEAL,ms=3,alpha=0.30,mec="none",label="$K=4$ commodities")
xx=np.array([m[g].min(),m[g].max()])
ax1.loglog(xx,np.exp(ic)*xx**sl,color=ORANGE,lw=2.2,label=f"fit  $t\\propto m^{{{sl:.2f}}}$")
ax1.loglog(xx,np.exp(ic)*xx[0]**sl*(xx/xx[0])**1.0,color=GREY,lw=1.1,ls="--",label="slope 1 (O($m$))")
ax1.set_xlabel("edges  $m$"); ax1.set_ylabel("NLF total wall-clock  (s)")
ax1.set_title(f"(A) $K=4$: $t\\propto m^{{{sl:.2f}}}$ over {int(g.sum())} real graphs",fontsize=9.6)
ax1.grid(alpha=0.25,which="both"); ax1.legend(fontsize=8,loc="upper left")
# --- (B) chord-step histogram ---
vals,cnts=np.unique(it,return_counts=True)
ax2.bar(vals,cnts,color=TEAL,width=0.8)
ax2.axvline(np.median(it),color=ORANGE,lw=1.6,label=f"median {int(np.median(it))} steps")
ax2.set_xlabel("Chord Newton steps"); ax2.set_ylabel("number of graphs")
ax2.set_title("(B) chord-Newton step count: flat in size and class",fontsize=9.6)
ax2.grid(alpha=0.25,axis="y"); ax2.legend(fontsize=8)
plt.tight_layout()
out=D+"nlf_mc_corpus.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
pe=t[g]/m[g]*1e6
print(f"wrote {out} | {int(g.sum())} graphs (m>=1e3), t~m^{sl:.3f}, per-edge median {np.median(pe):.2f} us "
      f"(p95 {np.percentile(pe,95):.1f}), steps {it.min()}-{it.max()} (median {int(np.median(it))}), "
      f"setups median {int(np.median(su))} max {su.max()}")
