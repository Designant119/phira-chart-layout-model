#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motif 级 MLP v2：节奏 + 局部中位 + 前4音相对轨迹 + 线宽统计 -> 12 音 x 残差。"""
import os,glob,json,time,argparse
import numpy as np,torch,torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
class Net(nn.Module):
    def __init__(self,d=104,h=320):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,12))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)
def load_part(fp):
    d=np.load(fp,allow_pickle=True);Q=d['Q'].astype(np.float32);X=d['X'].astype(np.float32);C=d['ctx'].astype(np.float32);P4=d['prev4'].astype(np.float32);SD=d['ctx_std'].astype(np.float32);MN=d['ctx_min'].astype(np.float32);MX=d['ctx_max'].astype(np.float32);fid=d['fid'].astype(str)
    F=np.concatenate([Q,C[:,None],P4-C[:,None],SD[:,None],(MN-C)[:,None],(MX-C)[:,None]],axis=1);Y=X-C[:,None]
    return F,Y,C,fid
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parts',required=True);ap.add_argument('--root',default='/home/user/Doubao/chats/38453392661469954/_run_real_traj');ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--bs',type=int,default=8192);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);split=json.load(open(os.path.join(a.root,'train_data_traj_fixed','split.json')));valf=set(split['val']);testf=set(split['test']);parts=sorted(glob.glob(os.path.join(a.parts,'part_*.npz')))
    def collect(fids):
        Fs=[];Ys=[];Cs=[]
        for fp in parts:
            F,Y,C,fid=load_part(fp);m=np.isin(fid,list(fids))
            if m.sum():Fs.append(F[m]);Ys.append(Y[m]);Cs.append(C[m])
        return np.concatenate(Fs),np.concatenate(Ys),np.concatenate(Cs)
    VF,VY,VC=collect(valf);TF,TY,TC=collect(testf);print('val',len(VF),'test',len(TF),'dim',VF.shape[1],flush=True)
    net=Net(VF.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5);loss_fn=nn.SmoothL1Loss(beta=0.05)
    for ep in range(a.epochs):
        t0=time.time();tot=0;n=0
        for fp in parts:
            F,Y,C,fid=load_part(fp);m=~np.isin(fid,list(valf|testf))
            if not m.any():continue
            dl=DataLoader(TensorDataset(torch.from_numpy(F[m]),torch.from_numpy(Y[m])),batch_size=a.bs,shuffle=True,drop_last=True)
            net.train()
            for xb,yb in dl:
                opt.zero_grad();loss=loss_fn(net(xb),yb);loss.backward();opt.step();tot+=float(loss.detach())*len(xb);n+=len(xb)
        net.eval()
        with torch.no_grad():
            vp=net(torch.from_numpy(VF)).numpy();vmae=float(np.mean(np.abs((VC[:,None]+vp)-(VC[:,None]+VY)))*675)
            tp=net(torch.from_numpy(TF)).numpy();tmae=float(np.mean(np.abs((TC[:,None]+tp)-(TC[:,None]+TY)))*675)
        print(f'epoch {ep+1} loss {tot/n:.5f} val_MAE {vmae:.1f}px test_MAE {tmae:.1f}px {time.time()-t0:.0f}s',flush=True)
        torch.save({'model':net.state_dict(),'d':VF.shape[1]},a.out)
    print('baseline val',float(np.mean(np.abs(VY))*675),'test',float(np.mean(np.abs(TY))*675),flush=True)
if __name__=='__main__':main()
