#!/usr/bin/env python3
"""Schematic (panel b of fig:cusp): pseudo-arclength predictor-corrector for the max-flow fold,
drawn in the scalar case where phi and alpha are both scalars, so the equilibrium set is a curve
in the (phi, alpha) plane. Shows the equilibrium curve rho(phi)=alpha d; a previous solution
(phi0,alpha0); the unit tangent (phidot,alphadot); the predictor step Ds along the tangent to an
off-curve point P*; the equilibrium residual r (vertical gap to the curve) and the arclength line
N=0 (perp to the tangent); and the Newton corrector (dphi,dalpha) as a vector-addition triangle
dphi = dalpha*u + w, with u=J^+ d (along the tangent) and w=J^+ r (horizontal, kills r)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

TEAL="#00435A"; AQUA="#31CBC8"; ORANGE="#E8743B"; GREY="#7A8A90"; INK="#12303c"; UBLUE="#1596a6"

Fstar=1.0
rho  = lambda f: 1-np.exp(-f)
drho = lambda f: np.exp(-f)
f0=0.55; a0=rho(f0); slope=drho(f0)
t=np.array([1.0,slope]); t/=np.linalg.norm(t)            # unit tangent (phidot,alphadot)
fd,ad=t
Ds=0.90
Pstar=np.array([f0,a0])+Ds*t                              # predictor (off the curve)
fst,ast=Pstar
J=drho(fst); r=ast-rho(fst)                               # equilibrium residual (vertical gap)
u=1.0/J; w=r/J                                            # u=J^+ d (tangent dir),  w=J^+ r (horizontal)
dalpha=-(fd*w)/(ad+fd*u)                                  # bordered block-elimination (N=0 at P*)
dphi=dalpha*u+w
P1=Pstar+np.array([dphi,dalpha])                          # corrected point (lands on the curve)
Qw=Pstar+np.array([w,0.0])                                # triangle: P* --w--> Qw --dalpha*u--> P1

fig,ax=plt.subplots(figsize=(4.9,3.55))
ff=np.linspace(0,2.8,400)
ax.plot(ff,rho(ff),color=TEAL,lw=2.6,zorder=3)
ax.axhline(Fstar,ls=(0,(6,4)),color=GREY,lw=1.3)
ax.text(2.30,Fstar+0.006,r"$F^*$ (max flow)",color=GREY,fontsize=10.5,ha="right",va="bottom")
ax.text(2.18,0.905,r"$\rho(\phi)=\alpha d$",
        color=TEAL,fontsize=10.3,ha="center",va="bottom",rotation=9)

# --- arclength constraint line N=0 through P*, perpendicular to the tangent ---
perp=np.array([-ad,fd]); L0=Pstar-0.34*perp; L1=Pstar+0.155*perp
ax.plot([L0[0],L1[0]],[L0[1],L1[1]],ls=(0,(4,3)),color=GREY,lw=1.4,zorder=2)  # N=0 arclength line (label removed)

# --- predictor: Ds along the unit tangent (the teal line from (phi0,alpha0) to P*) ---
ax.annotate("",xy=Pstar,xytext=(f0,a0),
            arrowprops=dict(arrowstyle="-|>",color=AQUA,lw=2.8,shrinkA=0,shrinkB=0),zorder=5)
ax.text(0.905,0.705,r"$\Delta s\,(\dot\phi,\dot\alpha)$",color="#128a99",fontsize=11,
        ha="center",va="center",rotation=49)

# --- residual r: vertical gap from P* down to the curve ---
ax.plot([fst,fst],[rho(fst),ast],ls=":",color=INK,lw=1.7,zorder=4)
ax.text(fst-0.028,(ast+rho(fst))/2,r"$r$",color=INK,fontsize=13,ha="right",va="center")

# --- corrector as a vector-addition triangle:  (dphi,dalpha) = w + dalpha*u ---
ax.annotate("",xy=Qw,xytext=tuple(Pstar),                 # leg 1: w (horizontal, kills r)
            arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=2.2,shrinkA=0,shrinkB=0),zorder=6)
ax.annotate("",xy=P1,xytext=tuple(Qw),                    # leg 2: dalpha*u (tangent direction)
            arrowprops=dict(arrowstyle="-|>",color=UBLUE,lw=2.2,ls=(0,(5,2)),shrinkA=0,shrinkB=0),zorder=6)
ax.annotate("",xy=P1,xytext=tuple(Pstar),                 # resultant: the corrector step
            arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=3.2,shrinkA=0,shrinkB=0),zorder=7)
ax.text((Pstar[0]+Qw[0])/2,Qw[1]+0.013,r"$w=J^{+}r$",color=ORANGE,fontsize=11,ha="center",va="bottom")
ax.text(1.70,0.792,r"$\delta\alpha\,u,\ \ u=J^{+}d$",color=UBLUE,fontsize=10.5,ha="left",va="top")
ax.text(Pstar[0]-0.055,(Pstar[1]+P1[1])/2+0.028,r"$(\delta\phi,\delta\alpha)$",
        color=ORANGE,fontsize=10.8,ha="right",va="center")

# --- points ---
ax.plot(f0,a0,"o",color=INK,ms=6.5,zorder=8)
ax.text(f0+0.10,a0+0.002,r"$(\phi_0,\alpha_0)$"+"\nprev. solution",color=INK,fontsize=10.3,ha="left",va="center")
ax.plot(*Pstar,"o",color=INK,ms=6.5,zorder=8)
ax.text(Pstar[0]+0.03,Pstar[1]+0.052,r"$P^*$",color=INK,fontsize=11,ha="left",va="bottom")
ax.plot(*P1,"o",color=INK,ms=6.5,zorder=8)
ax.text(P1[0]-0.04,P1[1]-0.028,"corrected",color=INK,fontsize=10.3,ha="right",va="top")

# --- relation box, in the empty lower-right (below the curve) ---
ax.text(0.63,0.405,
        "bordered corrector\n"
        r"$J\,\delta\phi-d\,\delta\alpha=r$"+"\n"
        r"$\dot\phi\,\delta\phi+\dot\alpha\,\delta\alpha=-N$"+"\n"
        r"$\Rightarrow\ \ \delta\phi=\delta\alpha\,u+w$",
        transform=ax.transAxes,fontsize=10.0,ha="left",va="top",linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4",fc="white",ec=GREY,lw=1.0,alpha=0.95))

ax.set_xlim(0.12,2.35); ax.set_ylim(0.33,1.055)
ax.set_xlabel(r"potential  $\phi$   (cut amplitude $\psi=\chi^\top\phi$)",fontsize=11)
ax.set_ylabel(r"load  $\alpha$",fontsize=11.5)
ax.tick_params(labelsize=9)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(pad=0.5)
out="/Users/oren/code/nlf/doc/arclength.pdf"
plt.savefig(out); plt.savefig(out.replace(".pdf",".png"),dpi=170)
print("wrote",out)
