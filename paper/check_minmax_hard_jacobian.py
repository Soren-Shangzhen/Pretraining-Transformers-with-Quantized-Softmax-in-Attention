"""P0-1 check, float64: derivative of the FULL MinMax hard-Nearest map (grid values depend on row extrema),
autograd vs closed form c_j w_j (1[i=argmin]-1[i=argmax]) vs central finite differences, for rows sampled
away from index boundaries and extrema ties. Also: 2-key row == exact softmax; FW hard map locally constant."""
import torch, itertools
torch.manual_seed(0); torch.set_num_threads(2)
K=4
def w_hard(s):
    m=s.max(); mu=s.min(); h=(m-mu)/K; zmin=mu-m; z=s-m
    rel=(z-zmin)/h; r=torch.floor(rel).clamp(0,K-1); t=((z-(zmin+r*h))/h).clamp(0,1)
    l=r+(t>=0.5).double(); return torch.exp(zmin+l*h), l
def ok(s):
    m=s.max(); mu=s.min(); h=(m-mu)/K; rel=(s-mu)/h; frac=rel-torch.floor(rel)
    srt=torch.sort(s).values
    interior=(s!=m)&(s!=mu)
    bad=(((frac-0.5).abs()<0.08)|(frac<0.08)|(frac>0.92))&interior
    return not bad.any() and (srt[1]-srt[0])>0.15 and (srt[-1]-srt[-2])>0.15
rows=0; e_cf=0.; e_fd=0.; tries=0
while rows<100 and tries<5000:
    tries+=1; n=6; s=torch.randn(n,dtype=torch.float64)*3
    if not ok(s): continue
    J=torch.autograd.functional.jacobian(lambda x:w_hard(x)[0], s)
    w,l=w_hard(s); imax=int(s.argmax()); imin=int(s.argmin()); c=1-l/K
    Jc=torch.zeros_like(J); Jc[:,imin]=c*w; Jc[:,imax]=-c*w
    e_cf=max(e_cf,float((J-Jc).abs().max()))
    eps=1e-6; Jfd=torch.zeros_like(J)
    for i in range(n):
        e=torch.zeros(n,dtype=torch.float64); e[i]=eps; Jfd[:,i]=(w_hard(s+e)[0]-w_hard(s-e)[0])/(2*eps)
    e_fd=max(e_fd,float((J-Jfd).abs().max())); rows+=1
print(f"MinMax hard map, K={K}: rows={rows} (tries {tries})  max|autograd-closedform|={e_cf:.2e}  max|autograd-FD|={e_fd:.2e}")
s=torch.tensor([0.3,-1.7],dtype=torch.float64)
P=(lambda w:w/w.sum())(w_hard(s)[0]); print("2-key row: P_hard=",P.numpy()," softmax=",torch.softmax(s,0).numpy())
Jp=torch.autograd.functional.jacobian(lambda x:(lambda w:w/w.sum())(w_hard(x)[0]), s); Je=torch.autograd.functional.jacobian(lambda x:torch.softmax(x,0), s)
print("2-key row: max|J_hardnorm - J_softmax| =",float((Jp-Je).abs().max()))
tau=6.0
def w_fw(s):
    m=s.max(); z=(s-m).clamp(-tau,0); h=tau/K; rel=(z+tau)/h; r=torch.floor(rel).clamp(0,K-1)
    t=((z-(-tau+r*h))/h).clamp(0,1); l=r+(t>=0.5).double(); w=torch.exp(-tau+l*h); return torch.where(s-m<-tau, torch.zeros_like(w), w)
s=torch.tensor([0.0,-1.2,-2.9,-4.4,-7.0],dtype=torch.float64)
J=torch.autograd.functional.jacobian(w_fw,s); print("FW hard map Jacobian max|.| =",float(J.abs().max()), "(locally constant when index and tail mask are fixed)")
