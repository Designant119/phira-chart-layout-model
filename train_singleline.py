#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单判定线因果 Transformer：看 128 音历史节奏与已生成 x，预测下一批 x。"""
import os,json,time,argparse,math
import numpy as np,torch,torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
class SeqNet(nn.Module):
    def __init__(self,d=96,layers=2,heads=4,nf=14):
        super().__init__();self.inp=nn.Linear(nf,d);self.pos=nn.Parameter(torch.zeros(1,128,d));nn.init.normal_(self.pos,std=.02)
        layer=nn.TransformerEncoderLayer(d,heads,4*d,dropout=.1,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,layers);self.out=nn.Linear(d,1)
    def forward(self,f,mask=None):
        h=self.inp(f)+self.pos[:,:f.shape[1]];h=self.enc(h,mask=mask);return self.out(h).squeeze(-1)
def ar_eval(net,F,Y,M,bs=256):
    net.eval();preds=[]
    with torch.no_grad():
        for i in range(0,len(F),bs):
            f=torch.from_numpy(F[i:i+bs].copy());y=torch.zeros(f.shape[0],128)
            for k in range(128):
                if k>=1:y[:,k]=y[:,k-1]
                f[:,k,8]=y[:,k-1] if k>=1 else 0.0
                f[:,k,9]=y[:,k-2] if k>=2 else 0.0
                f[:,k,10]=f[:,k,8]-f[:,k,9]
                out=net(f)[:,k];y[:,k]=out
            preds.append(y.numpy())
    P=np.concatenate(preds);m=M.astype(bool);return float(np.mean(np.abs(P[m]-Y[m]))*675)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--split',required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--bs',type=int,default=128);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--max-train',type=int,default=40000)
    a=ap.parse_args();torch.set_num_threads(2);d=np.load(a.data,allow_pickle=True);F=d['F'].astype(np.float32);Y=d['Y'].astype(np.float32);M=d['M'].astype(np.float32);fid=d['fid'].astype(str);split=json.load(open(a.split));val=set(split['val']);test=set(split['test']);tr=~np.isin(fid,list(val|test));va=np.isin(fid,list(val));te=np.isin(fid,list(test));tr_idx=np.where(tr)[0];rng=np.random.default_rng(0);tr_idx=rng.choice(tr_idx,size=min(a.max_train,len(tr_idx)),replace=False)
    print('windows',len(F),'train',len(tr_idx),'val',va.sum(),'test',te.sum(),flush=True)
    net=SeqNet();opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5);mask=torch.triu(torch.full((128,128),float('-inf')),diagonal=1)
    for ep in range(a.epochs):
        t0=time.time();net.train();dl=DataLoader(TensorDataset(torch.from_numpy(F[tr_idx]),torch.from_numpy(Y[tr_idx]),torch.from_numpy(M[tr_idx])),batch_size=a.bs,shuffle=True,drop_last=True);tot=0;n=0
        for f,y,m in dl:
            opt.zero_grad();out=net(f,mask=mask);loss=(torch.abs(out-y)*m).sum()/m.sum();loss.backward();opt.step();tot+=float(loss)*len(f);n+=len(f)
        vmae=ar_eval(net,F[va],Y[va],M[va]);tmae=ar_eval(net,F[te],Y[te],M[te])
        print(f'epoch {ep+1} loss {tot/n:.4f} val_AR_MAE {vmae:.1f}px test_AR_MAE {tmae:.1f}px {time.time()-t0:.0f}s',flush=True)
        torch.save({'model':net.state_dict()},a.out)
    base=float(np.mean(np.abs(Y[M.astype(bool)])*675));print('baseline',base,flush=True)
if __name__=='__main__':main()
