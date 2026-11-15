#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motif 级条件 MLP：节奏+局部中位+prev -> 12 音的 x 残差。"""
import os,glob,json,time,argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self,d=98,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,12))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)

def load_part(fp):
    d=np.load(fp,allow_pickle=True)
    Q=d['Q'].astype(np.float32); X=d['X'].astype(np.float32); C=d['ctx'].astype(np.float32); P=d['prev'].astype(np.float32)
    fid=d['fid'].astype(str)
    F=np.concatenate([Q,C[:,None],(P-C)[:,None]],axis=1)
    Y=X-C[:,None]
    return F,Y,C,fid

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parts',required=True); ap.add_argument('--root',default='/home/user/Doubao/chats/38453392661469954/_run_real_traj'); ap.add_argument('--epochs',type=int,default=4); ap.add_argument('--bs',type=int,default=8192); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--out',required=True)
    a=ap.parse_args(); torch.set_num_threads(2)
    split=json.load(open(os.path.join(a.root,'train_data_traj_fixed','split.json'))); valf=set(split['val']); testf=set(split['test'])
    parts=sorted(glob.glob(os.path.join(a.parts,'part_*.npz')));print('parts',len(parts))
    # 收集 val/test
    def collect(fids):
        Fs=[];Ys=[];Cs=[];fids_out=[]
        for fp in parts:
            F,Y,C,fid=load_part(fp);m=np.isin(fid,list(fids))
            if m.sum():Fs.append(F[m]);Ys.append(Y[m]);Cs.append(C[m]);fids_out+=list(fid[m])
        return np.concatenate(Fs),np.concatenate(Ys),np.concatenate(Cs)
    VF,VY,VC=collect(valf); TF,TY,TC=collect(testf);print('val',len(VF),'test',len(TF))
    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5); loss_fn=nn.SmoothL1Loss(beta=0.05)
    for ep in range(a.epochs):
        t0=time.time(); tot=0;n=0
        for fp in parts:
            F,Y,C,fid=load_part(fp); m=~np.isin(fid,list(valf|testf));
            if not m.any():continue
            F=F[m];Y=Y[m]
            ds=TensorDataset(torch.from_numpy(F),torch.from_numpy(Y));dl=DataLoader(ds,batch_size=a.bs,shuffle=True,drop_last=True)
            for xb,yb in dl:
                opt.zero_grad(); pred=net(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step()
                tot+=float(loss.detach())*len(xb); n+=len(xb)
        net.eval()
        with torch.no_grad():
            vp=net(torch.from_numpy(VF)).numpy(); vm=float(np.mean(np.abs(VC+vp[:,None]-np.load(parts[0])['X'][:0].shape[0]*0))) if False else 0
            # 绝对误差：ctx + yhat 对比真实 X；真实 X 需从 Y 反推 C+? Y 是 X-C，故真实 X=C+Y
            vtrue=VC[:,None]+VY; vmae=float(np.mean(np.abs((VC[:,None]+vp)-vtrue)))*675
            tp=net(torch.from_numpy(TF)).numpy(); ttrue=TC[:,None]+TY; tmae=float(np.mean(np.abs((TC[:,None]+tp)-ttrue)))*675
        print(f'epoch {ep+1} loss {tot/n:.5f} val_MAE {vmae:.1f}px test_MAE {tmae:.1f}px {time.time()-t0:.0f}s',flush=True)
        torch.save({'model':net.state_dict(),'d':98},a.out)
    # baseline
    print('baseline val',float(np.mean(np.abs(VY))*675),'test',float(np.mean(np.abs(TY))*675),flush=True)
if __name__=='__main__':main()
