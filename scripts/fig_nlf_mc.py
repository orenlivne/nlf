#!/usr/bin/env python3
"""Multicommodity illustrative figure (paper section): the K=3 heaviest REAL OD pairs of the Sioux
Falls demand matrix, solved jointly by NLF at rush-hour load. (A) per-commodity equilibrium flows
(parallel offsets, one color each); (B) the shared congestion they create (utilization of the
practical capacity); (C) the coupling it induces: per-edge utilization, joint equilibrium vs the
superposition of each commodity solved alone. Data: /tmp/siouxfalls_mc.csv (+_od.csv) + geojson."""
import csv, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
TEAL="#00435A"; ORANGE="#E8743B"; GREY="#8A99A0"
CK=["#0072B2","#D55E00","#009E73"]                                  # commodity colors (CB-safe)
D="/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/"
coords={}
gj=json.load(open("/tmp/tn_probe/SiouxFalls/SiouxFallsCoordinates.geojson"))
for ft in gj["features"]:
    coords[int(ft["properties"]["id"])]=tuple(ft["geometry"]["coordinates"])
rows=list(csv.DictReader(open("/tmp/siouxfalls_mc.csv")))
od=[(int(r["s"]),int(r["t"])) for r in csv.DictReader(open("/tmp/siouxfalls_mc_od.csv"))]
U=np.array([int(r["u"]) for r in rows]); V=np.array([int(r["v"]) for r in rows])
cap=np.array([float(r["cap"]) for r in rows])
F=np.array([[float(r[f"f{k}"]) for k in (1,2,3)] for r in rows])
S=np.array([[float(r[f"s{k}"]) for k in (1,2,3)] for r in rows])
utilJ=np.linalg.norm(F,axis=1)/cap; utilS=np.linalg.norm(S,axis=1)/cap
P1=np.array([coords[u] for u in U]); P2=np.array([coords[v] for v in V])
fig,axs=plt.subplots(1,3,figsize=(12.6,4.5))

def basemap(ax):
    for a,b in zip(P1,P2): ax.plot([a[0],b[0]],[a[1],b[1]],color="#D7DEE2",lw=1.0,zorder=1)
    ax.set_aspect("equal"); ax.axis("off")

# --- (A) per-commodity flows, parallel offsets ---
ax=axs[0]; basemap(ax)
span=max(P1[:,0].max()-P1[:,0].min(), P1[:,1].max()-P1[:,1].min())
off=0.011*span
for k in range(3):
    segs=[]; ws=[]
    for e in range(len(rows)):
        a=np.array(coords[U[e]]); b=np.array(coords[V[e]])
        d=b-a; L=np.hypot(*d)
        if L==0 or abs(F[e,k])/cap[e]<0.02: continue
        nrm=np.array([-d[1],d[0]])/L*(k-1)*off
        segs.append([a+nrm,b+nrm]); ws.append(0.6+7.0*(abs(F[e,k])/cap[e])/max(1.0,np.linalg.norm(F,axis=1).max()/cap.min()/2.2))
    ax.add_collection(LineCollection(segs,colors=CK[k],linewidths=ws,alpha=0.85,capstyle="round",zorder=2+k))
for k,(s_,t) in enumerate(od):
    ax.plot(*coords[s_],"o",ms=9.5,mfc="white",mec=CK[k],mew=2.0,zorder=9)
    ax.annotate(f"O$_{k+1}$",coords[s_],textcoords="offset points",xytext=(-15,4),fontsize=9,color=CK[k],fontweight="bold")
    ax.plot(*coords[t],"s",ms=9.5,mfc=CK[k],mec="k",mew=1.1,zorder=9)
    ax.annotate(f"D$_{k+1}$",coords[t],textcoords="offset points",xytext=(9,6),fontsize=9,color=CK[k],fontweight="bold")
ax.set_title("(A) three heaviest real OD demands,\njoint equilibrium flows",fontsize=9.6)

# --- (B) shared congestion ---
ax=axs[1]; basemap(ax)
segs=[[coords[U[e]],coords[V[e]]] for e in range(len(rows))]
unorm=plt.Normalize(0,max(1.25,utilJ.max()*1.02))
lc=LineCollection(segs,cmap="RdYlBu_r",norm=unorm,linewidths=1.0+6.0*utilJ/max(1.0,utilJ.max()/1.6),capstyle="round",zorder=2)
lc.set_array(utilJ); ax.add_collection(lc)
cb=plt.colorbar(lc,ax=ax,orientation="horizontal",fraction=0.05,pad=0.02,shrink=0.85)
cb.set_label(r"utilization  $\|\mathbf{f}_e\|/c_e$",fontsize=8); cb.ax.tick_params(labelsize=7)
for k,(s_,t) in enumerate(od):
    ax.plot(*coords[s_],"o",ms=7.5,mfc="white",mec=CK[k],mew=1.7,zorder=9)
    ax.plot(*coords[t],"s",ms=7.5,mfc=CK[k],mec="k",mew=1.0,zorder=9)
ax.set_title("(B) the congestion they share:\nover-saturated corridors",fontsize=9.6)

# --- (C) the congestion externality: shared load inflates each commodity's marginal cost ---
ax=axs[2]
t0=np.array([float(r["t0"]) for r in rows])
tt=lambda x: x+0.15*x**4                       # t(f) = t0 h(f/c), h(x)=x+0.15x^4 (regularized BPR)
k0=np.argmax([np.abs(F[:,k]).sum() for k in range(3)])   # heaviest commodity
act=np.abs(F[:,k0])/cap>0.05
infl=tt(np.linalg.norm(F[act],axis=1)/cap[act])/tt(np.abs(F[act,k0])/cap[act])
ax.axhline(1.0,color=GREY,lw=1.2,ls="--",zorder=1,label="no externality (alone)")
sc=ax.scatter(utilJ[act],infl,c=utilJ[act],cmap="RdYlBu_r",norm=unorm,s=36,ec="k",lw=0.4,zorder=3)
ax.set_yscale("log")
ax.set_xlabel(r"corridor utilization  $\|\mathbf{f}_e\|/c_e$  (joint)",fontsize=8.5)
ax.set_ylabel(f"commodity-{k0+1} marginal-cost inflation\n(joint / alone on the same edge)",fontsize=8.5)
mx=np.argmax(infl)
ax.annotate(f"$\\times${infl[mx]:.1f} on the worst\nshared corridor",(utilJ[act][mx],infl[mx]),
            textcoords="offset points",xytext=(12,-4),fontsize=8,color=TEAL,
            arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.1))
ax.grid(alpha=0.25,which="both"); ax.legend(fontsize=8,loc="lower right")
ax.set_title("(C) what the coupling does on real data:\nnot rerouting, but slowdown",fontsize=9.6)
plt.tight_layout()
out=D+"nlf_mc_sioux.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
dev=np.abs(utilJ-utilS); print(f"wrote {out} | max|joint-solo| util dev {dev.max():.3f} on {len(rows)} edges; "
      f"max util joint {utilJ.max():.2f} solo {utilS.max():.2f}")
