#!/usr/bin/env python3
"""Figure 1 (3x2): the three instances of the resistor-network framework, one row each.
Row 1 max-flow (tanh): (A) law/conductance/energy; (B) the saturated min-cut = active set.
Row 2 min-delay routing (Kleinrock 1/(c-f)): (A) law saturates at capacity; (B) the alpha(V) fold.
Row 3 congestion (BPR): (A) law unbounded, conductance bounded away from 0 -> no fold;
(B) REAL application: the Sioux Falls road network colored by the computed NLF equilibrium f/c.
Data: /tmp/siouxfalls_flow.csv (NLF solve) + /tmp/tn_probe/SiouxFalls/SiouxFallsCoordinates.geojson."""
import json, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

TEAL="#00435A"; AQUA="#31CBC8"; ORANGE="#E8743B"; RED="#C0392B"; GREY="#8A99A0"
fig, axs = plt.subplots(3, 2, figsize=(12, 10.8), gridspec_kw={"width_ratios":[1,1.18]})

def lawpanel(ax, g, f, dr, Psi, title, cap=None, caplab="$\\pm c_e$"):
    if cap is not None:
        ax.axhspan(-cap, cap, color=AQUA, alpha=0.10)
        ax.axhline(cap, ls=":", color=GREY, lw=1)
        ax.text(g[-1], cap+0.05, caplab, color=GREY, fontsize=9, ha="right")
    ax.plot(g, f,  color=TEAL,   lw=2.6, label=r"$f=\rho_e(g)$  (flow)")
    ax.plot(g, dr, color=ORANGE, lw=2.2, label=r"$\rho'_e$  (conductance)")
    ax.plot(g, Psi, color=AQUA,  lw=2.0, ls="--", label=r"$\Psi_e(g)$  (scaled)")
    ax.set_xlabel("$g=(B^{\\top}\\phi)_e$"); ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)

# ---- Row 1 (A): max-flow tanh ----
g = np.linspace(-4, 4, 400); c = 1.0
lawpanel(axs[0,0], g, c*np.tanh(g/c), 1/np.cosh(g/c)**2, c**2*np.log(np.cosh(g/c))/3,
         "(A1) max-flow: saturating law, $\\rho'\\to0$ at the cut", cap=c)
axs[0,0].annotate("saturated: $\\rho'\\!\\to\\!0$, $f\\!\\to\\!c_e$", xy=(3.0,0.99), xytext=(0.8,1.45),
                  fontsize=8.5, color=RED, ha="center", arrowprops=dict(arrowstyle="->",color=RED,lw=1))
axs[0,0].set_ylim(-1.4, 1.85)

# ---- Row 1 (B): active-set / cut diagram ----
axB = axs[0,1]; axB.axis("off"); axB.set_xlim(-0.5,6.5); axB.set_ylim(-0.35,3.3)
pos = {"s":(0,1.5), "a":(2,2.4),"b":(2,0.6), "c":(4,2.4),"d":(4,0.6), "t":(6,1.5)}
phi = {"s":1.0,"a":0.92,"b":0.90,"c":0.12,"d":0.10,"t":0.0}
for k,(x,y) in pos.items():
    axB.scatter([x],[y], s=560, color=cm.coolwarm(1-phi[k]), edgecolors="k", zorder=3, linewidths=1.2)
    lab = {"s":"$s$","t":"$t$"}.get(k, "")
    axB.text(x,y, lab, ha="center", va="center", fontsize=12, fontweight="bold", zorder=4,
             color="white" if k in ("s","t") else "k")
edges = [("s","a",0),("s","b",0),("a","c",1),("b","c",1),("b","d",1),("c","t",0),("d","t",0)]
for (u,v,sat) in edges:
    x1,y1=pos[u]; x2,y2=pos[v]
    axB.annotate("", xy=(x2,y2), xytext=(x1,y1),
                 arrowprops=dict(arrowstyle="-|>", color=(RED if sat else TEAL),
                                 lw=(3.4 if sat else 2.0), shrinkA=15, shrinkB=15, alpha=0.9))
axB.plot([3,3],[0.0,3.0], ls=(0,(6,4)), color=RED, lw=1.6)
axB.text(3,3.12,"min cut = active set", color=RED, fontsize=9.5, ha="center", fontweight="bold")
axB.text(3,-0.25,"saturated edges carry full flow $f_e=c_e$ ($\\rho'\\!\\to\\!0$); all $s\\!\\to\\!t$ flow crosses the cut",
         color=RED, fontsize=8.3, ha="center")
axB.set_title("(B1) max-flow: the cut is the saturated edge set", fontsize=10.5)

# ---- Row 2 (A): Kleinrock min-delay 1/(c-f) ----
c2 = 1.0
g2 = np.linspace(1.0/c2+1e-3, 12, 400)             # g = t(f) = 1/(c-f) >= 1/c
f2 = c2 - 1.0/g2
dr2 = 1.0/g2**2
Psi2 = (c2*(g2-1.0/c2) - np.log(g2*c2))/3
lawpanel(axs[1,0], g2, f2, dr2, Psi2,
         "(A2) min-delay (Kleinrock $1/(c_e\\!-\\!f)$): saturates at capacity", cap=c2, caplab="$c_e$")
axs[1,0].set_ylim(-0.1, 1.5)
# ---- Row 2 (B): REAL application -- Abilene/Internet2 backbone, min-delay equilibrium ----
# Kleinrock law: marginal delay t(f)=1/(c-f) => rho(g)=c-1/g (g>=1/c, else 0), rho'(g)=1/g^2.
# Tiny Newton with load ramp (11 nodes); NLF's fold-class instance computed exactly.
AB = {"SEA":(-122.3,47.6),"SNV":(-122.0,37.4),"LA":(-118.2,34.0),"DEN":(-105.0,39.7),
      "KC":(-94.6,39.1),"HOU":(-95.4,29.8),"CHI":(-87.6,41.9),"IND":(-86.2,39.8),
      "ATL":(-84.4,33.7),"WAS":(-77.0,38.9),"NYC":(-74.0,40.7)}
LINKS=[("SEA","SNV"),("SEA","DEN"),("SNV","LA"),("SNV","DEN"),("LA","HOU"),("DEN","KC"),
       ("KC","HOU"),("KC","IND"),("HOU","ATL"),("CHI","IND"),("CHI","NYC"),("IND","ATL"),
       ("ATL","WAS"),("NYC","WAS")]
names=list(AB); idx={k:i for i,k in enumerate(names)}; nA=len(names); mA=len(LINKS)
Bm=np.zeros((nA,mA))
for e,(u,v) in enumerate(LINKS): Bm[idx[u],e]=-1; Bm[idx[v],e]=1
cap=np.ones(mA)                                   # unit capacities (10 Gb/s links)
dvec=np.zeros(nA); dvec[idx["SEA"]]=-1; dvec[idx["NYC"]]=1
def rho_dr(g):
    f=np.where(g>1.0/cap, cap-1.0/np.maximum(g,1e-12), 0.0)
    dr=np.where(g>1.0/cap, 1.0/np.maximum(g,1e-12)**2, 1e-6)
    return f,dr
phiA=np.zeros(nA)
for aload in np.linspace(0.15,1.65,9):            # load ramp to a congested operating point
    for _ in range(60):
        gA=Bm.T@phiA; fA,drA=rho_dr(gA); r=Bm@fA-aload*dvec
        if np.linalg.norm(r)<1e-12: break
        J=Bm@np.diag(drA)@Bm.T+1e-12*np.eye(nA)
        step=np.linalg.lstsq(J,-r,rcond=None)[0]; step-=step.mean()
        t=1.0
        for _ in range(40):
            ph=phiA+t*step; ph-=ph.mean(); fT,_=rho_dr(Bm.T@ph)
            if np.linalg.norm(Bm@fT-aload*dvec)<=np.linalg.norm(r): phiA=ph; break
            t*=0.5
gA=Bm.T@phiA; fA,_=rho_dr(gA); utilA=np.abs(fA)/cap
axF = axs[1,1]
normA=Normalize(vmin=0,vmax=1.0); cmapA=cm.get_cmap("RdYlBu_r")
for e,(u,v) in enumerate(LINKS):
    (x1,y1),(x2,y2)=AB[u],AB[v]
    axF.plot([x1,x2],[y1,y2],color=cmapA(normA(utilA[e])),lw=1.5+4.5*normA(utilA[e]),
             zorder=2,solid_capstyle="round")
for k,(x,y) in AB.items():
    axF.scatter([x],[y],s=42,color=TEAL,zorder=3)
    dy = -1.1 if k in ("SNV","HOU","ATL","WAS") else 0.65
    axF.text(x,y+dy,k,fontsize=7.5,ha="center",color=TEAL)
for nid,lab in (("SEA","$s$"),("NYC","$t$")):
    axF.annotate(lab,AB[nid],textcoords="offset points",xytext=(8,7),fontsize=12,
                 fontweight="bold",color=TEAL)
smA=cm.ScalarMappable(norm=normA,cmap=cmapA); smA.set_array([])
cbA=plt.colorbar(smA,ax=axF,orientation="horizontal",fraction=0.05,pad=0.10)
cbA.set_label("link utilization  $f_e/c_e$  (delay $=1/(c_e\\!-\\!f_e)$)",fontsize=8.5)
axF.set_title("(B2) min-delay: Abilene backbone, computed NLF equilibrium",fontsize=10.5)
axF.set_aspect(1.2); axF.set_anchor("C"); axF.axis("off")

# ---- Row 3 (A): BPR ----
c3, b3, p3 = 1.0, 0.15, 4.0
fb = np.linspace(0, 2.2, 400)
tb = 1.0*(1 + b3*(fb/c3)**p3)                       # t(f), t0=1
gb = tb*0 + tb                                       # g = t(f)
drb = 1.0/(1 + b3*p3*(fb/c3)**(p3-1))               # rho'(g) = 1/t'(f)... t'(f)=t0*b*p*f^{p-1}/c^p + r? (strict law)
# strictly convex form used in the solver: t(f) = r f + k f^p (r=t0/c, k=b t0/c), c=t0=1
r3, k3 = 1.0, 0.15
tb = r3*fb + k3*fb**p3
drb = 1.0/(r3 + k3*p3*fb**(p3-1))
Psib = (0.5*r3*fb**2 + k3/(p3+1)*fb**(p3+1))/2
axb = axs[2,0]
axb.plot(tb, fb,  color=TEAL,   lw=2.6, label=r"$f=\rho_e(g)$  (unbounded)")
axb.plot(tb, drb, color=ORANGE, lw=2.2, label=r"$\rho'_e\in(0,\rho'_{\max}]$")
axb.plot(tb, Psib, color=AQUA,  lw=2.0, ls="--", label=r"$\Psi_e(g)$  (scaled)")
axb.set_xlabel("$g=(B^{\\top}\\phi)_e$")
axb.set_title("(A3) congestion (BPR): no cap, conductance bounded $\\Rightarrow$ no fold", fontsize=10.5)
axb.legend(fontsize=8, loc="upper left"); axb.grid(alpha=0.25); axb.set_xlim(0, max(tb))

# ---- Row 3 (B): REAL Sioux Falls network, NLF equilibrium utilization ----
axR = axs[2,1]
coords = {}
gj = json.load(open("/tmp/tn_probe/SiouxFalls/SiouxFallsCoordinates.geojson"))
for ft in gj["features"]:
    coords[int(ft["properties"]["id"])] = tuple(ft["geometry"]["coordinates"])
rows = list(csv.DictReader(open("/tmp/siouxfalls_flow.csv")))
norm = Normalize(vmin=0, vmax=max(float(r["util"]) for r in rows))
cmap = cm.get_cmap("RdYlBu_r")
for r in rows:
    u,v,ut = int(r["u"]), int(r["v"]), float(r["util"])
    (x1,y1),(x2,y2) = coords[u], coords[v]
    axR.plot([x1,x2],[y1,y2], color=cmap(norm(ut)), lw=1.2+3.4*norm(ut), zorder=2,
             solid_capstyle="round")
xs=[coords[i][0] for i in coords]; ys=[coords[i][1] for i in coords]
axR.scatter(xs, ys, s=22, color=TEAL, zorder=3)
for nid,lab in ((19,"$s$"),(1,"$t$")):
    axR.annotate(lab, coords[nid], textcoords="offset points", xytext=(7,4),
                 fontsize=12, fontweight="bold", color=TEAL)
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = plt.colorbar(sm, ax=axR, orientation="horizontal", fraction=0.05, pad=0.04)
cb.set_label("equilibrium utilization  $|f_e|/c_e$", fontsize=8.5)
axR.set_title("(B3) congestion: Sioux Falls, computed NLF equilibrium", fontsize=10.5)
axR.set_aspect("equal"); axR.set_anchor("C"); axR.axis("off")

plt.tight_layout()
out = "/Users/oren/code/mg/maxflow/LAMG.jl/doc/paper_program/nlf_formulation.pdf"
plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"), dpi=140, bbox_inches="tight")
print("wrote", out)
