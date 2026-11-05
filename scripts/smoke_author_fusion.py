import sys
sys.path.insert(0,'scripts')
from extract_original_radar import FastHTTPFile
import h5py,numpy as np,torch
from torch import nn
from milliego.model import RadarOnlyMilliEgo,KerasLSTM
from pathlib import Path
class Fusion(RadarOnlyMilliEgo):
 def __init__(self):
  super().__init__()
  self.imu=KerasLSTM(6,128)
  self.imask=nn.Sequential(nn.Linear(2560,128,bias=False),nn.ReLU(),nn.Linear(128,2560,bias=False),nn.Sigmoid())
  def cross(a,b):return nn.Sequential(nn.Linear(a,128,bias=False),nn.ReLU(),nn.Linear(128,b,bias=False),nn.ReLU())
  self.i2r=cross(2560,4096);self.r2i=cross(4096,2560)
  self.lstm1=KerasLSTM(6656,512,.25)
 def forward_fused(self,z,imu):
  r=(z*self.mask(z.mean((-2,-1)))[:,:,None,None]).permute(0,2,3,1).flatten(1)
  i=self.imu(imu).flatten(1);i=i*self.imask(i)
  v=torch.cat([r*self.i2r(i),i*self.r2i(r)],1)[:,None]
  v=self.lstm2(self.lstm1(v))
  return torch.cat([self.translation(v),self.rotation(v)],-1)[:,0]
with h5py.File('data/turtle_seq_52.h5') as f:
 imu=f['imu_data'][1:77,:20]
 d={'mmwave_middle_data':f['mmwave_middle_data'][:77],'label_data':f['label_data'][:,:76]}
assert imu.shape==(76,20,6) and np.isfinite(imu).all()
a=d['mmwave_middle_data'][:,0,:,:,0]
x=torch.from_numpy(np.concatenate([np.repeat(a[:-1,None],3,1),np.repeat(a[1:,None],3,1)],1)).cuda()
y=torch.from_numpy(d['label_data'][0].astype('f4')).cuda()
imu=torch.from_numpy(imu.astype('f4')).cuda()
for cls in [Fusion,RadarOnlyMilliEgo]:
 torch.manual_seed(42)
 m=cls().cuda();m.load_cnn('weights/cnn_torch.pt')
 with torch.no_grad():z=torch.cat([m.cnn(b) for b in x.split(16)])
 def pred(sl):return m.forward_fused(z[sl],imu[sl]) if cls is Fusion else m.forward_features(z[sl,None])[:,0]
 def loss(p,y):return (p[:,:3]-y[:,:3]).square().mean()+.001*(p[:,3:]-y[:,3:]).square().mean()
 opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-5,alpha=.9,eps=1e-8)
 for _ in range(5):
  m.train()
  for start in (0,16,32):
   sl=slice(start,start+16);opt.zero_grad();l=loss(pred(sl),y[sl]);assert torch.isfinite(l);l.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step()
 m.eval()
 with torch.no_grad():
  p=pred(slice(49,None));g=y[49:];assert p.shape==g.shape and torch.isfinite(p).all()
  print(cls.__name__,torch.__version__,'updates=15 heldout_pairs=27','translation_m',float((p[:,:3]-g[:,:3]).square().sum(1).mean().sqrt()),'angle_components_deg',float((p[:,3:]-g[:,3:]).square().mean().sqrt()),flush=True)

