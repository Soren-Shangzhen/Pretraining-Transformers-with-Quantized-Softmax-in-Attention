import sys, torch
sys.path.insert(0,KI_REPO + '/analysis_main_v1/supplement_four_runs_20260923/code/checks')
from sup_common import load_root
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

qa,_,_=load_root('old_phaseB_rr')
g=torch.Generator().manual_seed(1)
att=torch.randn(1,4,256,256,generator=g)*(4/5)
T=256; 
for dev in ['cpu','cuda']:
    a=att.to(dev); causal=torch.ones(T,T,dtype=torch.bool,device=dev).tril()
    tau=6.0;K=4;h=tau/K
    grid=torch.arange(K+1,device=dev,dtype=a.dtype); bw=torch.exp(-tau+grid*h)
    print(dev,'boundary_w bits',[hex(x) for x in bw.cpu().view(torch.int32).tolist()])
    wold=qa._relevant_zero_reconstruct(a,causal,K,'nearest',tau=tau,range_grad='full')[0]
    wl=qa._relevant_zero_reconstruct(a,causal,K,'lerp',tau=tau,range_grad='full')[0]
    # direct: reuse the exact same intermediate via the nearest choose rule
    print(dev,'wold unique values in first mismatch search...')
    globals()[dev]=(wold.cpu(),wl.cpu())
wc,lc=globals()['cpu']; wg,lg=globals()['cuda']
print('w_lerp cpu==gpu', torch.equal(lc,lg), int((lc.view(torch.int32)!=lg.view(torch.int32)).sum()))
print('w_old cpu==gpu', torch.equal(wc,wg), int((wc.view(torch.int32)!=wg.view(torch.int32)).sum()))
vals=torch.unique(wc[wc>0]); print('cpu w_old distinct', vals.numel(), [hex(x) for x in vals[:8].view(torch.int32).tolist()])
vals=torch.unique(wg[wg>0]); print('gpu w_old distinct', vals.numel(), [hex(x) for x in vals[:8].view(torch.int32).tolist()])
