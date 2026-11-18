#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单判定线整窗直接生成：给 128 音的节奏，一次性输出 128 个 x，不做逐步自回归。"""
import os,json,time,argparse
import numpy as np,torch,torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
class WinNet(nn.Module):
    def __init__(self,d=96,layers=2,heads=4):
        super().__init__();self.inp=nn.Linear(8,d);self.pos=nn.Parameter(torch.zeros(1,128,d));nn.init.normal_(self.pos,std=.02)
        layer=nn.TransformerEncoderLayer(d,heads,4*d,dropout=.1,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,layers);self.out=nn.Linear(d,1)
    def forward(self,f):return self.out(self.enc(self.inp(f)+self.pos[:,:f.shape[1]])).squeeze(-1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--split',required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--bs',type=int,default=128);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--max-train',type=int,default=40000)
    a=ap.parse_args();torch.set_num_threads(2);d=np.load(a.data,allow_pickle=True);F=d['F'].astype(np.float32)[:,:,:8];Y=d['Y'].astype(np.float32);M=d['M'].astype(np.float32);fid=d['fid'].astype(str);split=json.load(open(a.split));val=set(split['val']);test=set(split['test']);tr=~np.isin(fid,list(val|test));va=np.isin(fid,list(val));te=np.isin(fid,list(test));idx=np.where(tr)[0];rng=np.random.default_rng(0);idx=rng.choice(idx,size=min(a.max_train,len(idx)),replace=False)
    print('windows',len(F),'train',len(idx),'val',va.sum(),'test',te.sum(),flush=True)
    net=WinNet();opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5);best=1e9
    for ep in range(a.epochs):
        t0=time.time();net.train();dl=DataLoader(TensorDataset(torch.from_numpy(F[idx]),torch.from_numpy(Y[idx]),torch.from_numpy(M[idx])),batch_size=a.bs,shuffle=True,drop_last=True);tot=0;n=0
        for f,y,m in dl:
            opt.zero_grad();out=net(f);loss=(torch.abs(out-y)*m).sum()/m.sum();loss.backward();opt.step();tot+=float(loss.detach())*len(f);n+=len(f)
        net.eval()
        with torch.no_grad():
            vm=net(torch.from_numpy(F[va])).numpy();vmae=float(np.mean(np.abs(vm[M[va].astype(bool)]-Y[va][M[va].astype(bool)])*675))
            tm=net(torch.from_numpy(F[te])).numpy();tmae=float(np.mean(np.abs(tm[M[te].astype(bool)]-Y[te][M[te].astype(bool)])*675))
        print(f'epoch {ep+1} loss {tot/n:.4f} val_MAE {vmae:.1f}px test_MAE {tmae:.1f}px {time.time()-t0:.0f}s',flush=True)
        if vmae<best:best=vmae;torch.save({'model':net.state_dict()},a.out);print('saved best',best,flush=True)
    print('baseline',float(np.mean(np.abs(Y[M.astype(bool)])*675)),flush=True)
if __name__=='__main__':main()
