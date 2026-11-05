from pathlib import Path
src=Path('runs/adc_heatmap_pilot_v1/expanded_pilot.py').read_text()
exec(src[:src.index('results={}')])
img=imgs['adc']; fit=img[np.unique(pairs[:nt])]
img=((img-fit.mean())/fit.std()).cuda()
def make(p):
 return img[p].repeat_interleave(3,dim=1)[:,None]
def label(p):
 d=np.linalg.inv(world[p[:,0]])@world[p[:,1]]
 return np.column_stack([d[:,:3,3],Rotation.from_matrix(d[:,:3,:3]).as_euler('XYZ',degrees=True)]).astype(np.float32),d
def join(p):
 assert np.all(p[:-1,1]==p[1:,0])
 return p.reshape(-1,2,2)[:,[0,1],[0,1]]
raw_train=pairs[:nt]
k=np.flatnonzero(raw_train[:-1,1]==raw_train[1:,0])
long_train=np.column_stack([raw_train[k,0],raw_train[k+1,1]])
short_test=pairs[nt+nv:][:228]
long_test=join(short_test)
truth,dt=label(long_test)
testd=dict(pairs=long_test,world_radar=world,deltas=dt,timestamps=ts)
print('train eligible',len(raw_train),len(long_train),'matched test',len(long_test),'dt',np.median(np.diff(ts[long_test],axis=1)),flush=True)
results=[]
for seed in (20261102,20261103,20261104):
 for arm in ('native','gap2'):
  torch.manual_seed(seed)
  m=RadarOnlyMilliEgo().cuda()
  opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
  tp=raw_train[k] if arm=='native' else long_train
  x=make(tp); y=torch.from_numpy(label(tp)[0]).cuda()
  m.train()
  for _ in range(1120):
   ix=torch.randint(len(x),(24,),device='cuda')
   p=m(x[ix])[:,0]; l=loss(p,y[ix]); assert torch.isfinite(l)
   opt.zero_grad(set_to_none=True);l.backward();opt.step()
  m.eval()
  with torch.inference_mode(): p=torch.cat([m(b)[:,0].cpu() for b in make(short_test if arm=='native' else long_test).split(32)]).numpy()
  if arm=='native':
   T=np.repeat(np.eye(4)[None],len(p),axis=0)
   T[:,:3,:3]=Rotation.from_euler('XYZ',p[:,3:],degrees=True).as_matrix();T[:,:3,3]=p[:,:3]
   T=T[::2]@T[1::2]
   p=np.column_stack([T[:,:3,3],Rotation.from_matrix(T[:,:3,:3]).as_euler('XYZ',degrees=True)])
  met,_=metrics(p,testd,np.arange(len(p)),'XYZ')
  a=truth[:,5];b=p[:,5];mask=abs(a)>=4
  row=dict(seed=seed,arm=arm,yaw=float(np.sqrt(np.mean((a-b)**2))),zero=float(np.sqrt(np.mean(a*a))),correct=int((np.sign(a[mask])==np.sign(b[mask])).sum()),turns=int(mask.sum()),ate=met['ate_initial_rmse_m'],path=met['segments'][0]['path_length_m'])
  print(json.dumps(row),flush=True);results.append(row)
(out/'gap2_diagnostic.json').write_text(json.dumps(results,indent=2))

