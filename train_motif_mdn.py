#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motif 混合密度模型：节奏/局部轨迹/整条线宽度 -> 多种 12 音排法分布。"""
import os,glob,json,time,argparse,math
import numpy as np,torch,torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
K,D=8,12
class MDN(nn.Module):
    def __init__(self,d=106,h=384):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,h),nn.GELU())
        self.pi=nn.Linear(h,K); self.mu=nn.Linear(h,K*D); self.ls=nn.Linear(h,K*D)
        nn.init.zeros_(self.ls.weight); nn.init.zeros_(self.ls.bias)
    def forward(self,x):
        h=self.trunk(x); pi=self.pi(h); mu=self.mu(h).view(-1,K,D); ls=self.ls(h).view(-1,K,D)
        return pi,mu,ls.clamp(-3.5,1.5)
def load_part(fp,sty):
    d=np.load(fp,allow_pickle=True);Q=d['Q'].astype(np.float32);X=d['X'].astype(np.float32);C=d['ctx'].astype(np.float32);P4=d['prev4'].astype(np.float32);SD=d['ctx_std'].astype(np.float32);MN=d['ctx_min'].astype(np.float32);MX=d['ctx_max'].astype(np.float32);fid=d['fid'].astype(str);line=d['line'].astype(np.int32)
    lm=np.zeros(len(Q),np.float32);ls=np.zeros(len(Q),np.float32)
    for i,(f,l) in enumerate(zip(fid,line)):
        v=sty.get(f,{}).get(str(int(l)))
        if v: lm[i]=v[0]; ls[i]=v[1]
    F=np.concatenate([Q,C[:,None],P4-C[:,None],SD[:,None],(MN-C)[:,None],(MX-C)[:,None],lm[:,None],ls[:,None]],axis=1)
    Y=X-C[:,None]
    return F,Y,C,fid
def nll(pi,mu,ls,y):
    logpi=torch.log_softmax(pi,1); sig=torch.exp(ls)
    lp=-0.5*(((y[:,None,:]-mu)/sig)**2 + 2*ls + math.log(2*math.pi)).sum(2)
    return -torch.logsumexp(logpi+lp,1).mean()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parts',required=True);ap.add_argument('--style',required=True);ap.add_argument('--root',default='/home/user/Doubao/chats/38453392661469954/_run_real_traj');ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--bs',type=int,default=8192);ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);sty=json.load(open(a.style));split=json.load(open(os.path.join(a.root,'train_data_traj_fixed','split.json')));valf=set(split['val']);testf=set(split['test']);parts=sorted(glob.glob(os.path.join(a.parts,'part_*.npz')))
    def collect(fids):
        Fs=[];Ys=[];Cs=[]
        for fp in parts:
            F,Y,C,fid=load_part(fp,sty);m=np.isin(fid,list(fids))
            if m.sum():Fs.append(F[m]);Ys.append(Y[m]);Cs.append(C[m])
        return np.concatenate(Fs),np.concatenate(Ys),np.concatenate(Cs)
    VF,VY,VC=collect(valf);TF,TY,TC=collect(testf);print('val',len(VF),'test',len(TF),'dim',VF.shape[1],flush=True)
    net=MDN(VF.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5)
    for ep in range(a.epochs):
        t0=time.time();tot=0;n=0
        for fp in parts:
            F,Y,C,fid=load_part(fp,sty);m=~np.isin(fid,list(valf|testf))
            if not m.any():continue
            dl=DataLoader(TensorDataset(torch.from_numpy(F[m]),torch.from_numpy(Y[m])),batch_size=a.bs,shuffle=True,drop_last=True);net.train()
            for xb,yb in dl:
                opt.zero_grad();pi,mu,ls=net(xb);loss=nll(pi,mu,ls,yb);loss.backward();opt.step();tot+=float(loss.detach())*len(xb);n+=len(xb)
        net.eval()
        with torch.no_grad():
            vp,vm,vl=net(torch.from_numpy(VF));vpi=vp.argmax(1);vpred=vm[torch.arange(len(vm)),vpi].numpy();vmae=float(np.mean(np.abs(vpred-VY)))*675
            tp,tm,tl=net(torch.from_numpy(TF));tpi=tp.argmax(1);tpred=tm[torch.arange(len(tm)),tpi].numpy();tmae=float(np.mean(np.abs(tpred-TY)))*675
        print(f'epoch {ep+1} nll {tot/n:.4f} val_MAE {vmae:.1f}px test_MAE {tmae:.1f}px {time.time()-t0:.0f}s',flush=True)
        torch.save({'model':net.state_dict(),'d':VF.shape[1],'K':K,'D':D},a.out)
    print('baseline val',float(np.mean(np.abs(VY))*675),'test',float(np.mean(np.abs(TY))*675),flush=True)
if __name__=='__main__':main()
