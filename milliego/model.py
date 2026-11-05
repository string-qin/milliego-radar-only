"""Radar-only adaptation of official milliEgo networks.py, with Keras 2.1.6 LSTM equations."""
import torch
from torch import nn
from torch.nn import functional as F
class SameConv(nn.Module):
    def __init__(self,ci,co,k,s):
        super().__init__();self.conv=nn.Conv2d(ci,co,k,stride=s,padding=0);self.k=k;self.s=s
    def forward(self,x):
        h,w=x.shape[-2:];ph=max(0,((h+self.s-1)//self.s-1)*self.s+self.k-h);pw=max(0,((w+self.s-1)//self.s-1)*self.s+self.k-w)
        return F.leaky_relu(self.conv(F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2))),.1)
class KerasLSTM(nn.Module):
    """IFCO order, hard_sigmoid gates, tanh cell, unit forget bias, per-gate time-constant dropout."""
    def __init__(self,inputs,hidden,dropout=0):
        super().__init__();self.hidden=hidden;self.dropout=dropout
        self.kernel=nn.Parameter(torch.empty(4*hidden,inputs))
        self.recurrent=nn.Parameter(torch.empty(4*hidden,hidden))
        self.bias=nn.Parameter(torch.zeros(4*hidden))
        nn.init.xavier_uniform_(self.kernel);nn.init.orthogonal_(self.recurrent)
        with torch.no_grad():self.bias[hidden:2*hidden].fill_(1)
    def forward(self,x):
        b,t,_=x.shape;h=x.new_zeros(b,self.hidden);c=h.clone();outputs=[]
        masks=[F.dropout(torch.ones_like(x[:,0]),self.dropout,True) for _ in range(4)] if self.training and self.dropout else None
        weights=self.kernel.chunk(4,0)
        for k in range(t):
            z=torch.cat([F.linear(x[:,k]*m,w) for m,w in zip(masks,weights)],-1) if masks else F.linear(x[:,k],self.kernel)
            z=z+F.linear(h,self.recurrent,self.bias);i,f,g,o=z.chunk(4,-1)
            i=(.2*i+.5).clamp(0,1);f=(.2*f+.5).clamp(0,1);o=(.2*o+.5).clamp(0,1)
            c=f*c+i*torch.tanh(g);h=o*torch.tanh(c);outputs.append(h)
        return torch.stack(outputs,1)
class RadarOnlyMilliEgo(nn.Module):
    def __init__(self):
        super().__init__();dims=[(6,64,7,2),(64,128,5,2),(128,256,5,2),(256,256,3,1),(256,512,3,2),(512,512,3,1),(512,512,3,2),(512,512,3,1),(512,1024,3,2)]
        self.cnn=nn.Sequential(*(SameConv(*d) for d in dims))
        self.mask=nn.Sequential(nn.Linear(1024,4,bias=False),nn.ReLU(),nn.Linear(4,1024,bias=False),nn.Sigmoid())
        self.lstm1=KerasLSTM(4096,512,.25);self.lstm2=KerasLSTM(512,512)
        def head():return nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(.25),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
        self.translation=head();self.rotation=head()
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
    def load_cnn(self,path):
        state=torch.load(path,weights_only=True,map_location="cpu")["cnn_state_dict"]
        self.cnn.load_state_dict(state,strict=True)
        self.cnn.requires_grad_(False)
    def forward_features(self,z):
        # z: batch,time,channel,height,width, preserving official NHWC Flatten order
        b,t,c,h,w=z.shape
        gate=self.mask(z.mean((-2,-1)))
        z=(z*gate[...,None,None]).permute(0,1,3,4,2).reshape(b,t,4096)
        z=self.lstm2(self.lstm1(z))
        return torch.cat((self.translation(z),self.rotation(z)),-1)
    def forward(self,x):
        b,t,c,h,w=x.shape;z=self.cnn(x.reshape(b*t,c,h,w))
        return self.forward_features(z.reshape(b,t,*z.shape[1:]))

