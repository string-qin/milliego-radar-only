from pathlib import Path
g=Path('runs/adc_heatmap_pilot_v1/gap2_diagnostic.py').read_text()
exec(g[:g.index('results=[]')])
from time import monotonic
from scripts.validate_projection_diagnosis import save_tum
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pilot=out
holdout=pilot/'lab_run1_gap2_holdout'
holdout.mkdir(exist_ok=True)
seq=seq.with_name('12_21_2020_arpg_lab_run1')
adc=seq/'single_chip/adc_samples'
times=np.loadtxt(adc/'timestamps.txt')
gt_t=np.loadtxt(seq/'groundtruth/timestamps.txt')
gt_pose=np.loadtxt(seq/'groundtruth/groundtruth_poses.txt')
j=np.searchsorted(gt_t,times).clip(1,len(gt_t)-1)
valid=np.flatnonzero((times>=gt_t[0])&(times<=gt_t[-1])&(gt_t[j]-gt_t[j-1]<=.2))
cuts=np.flatnonzero((np.diff(valid)!=1)|(np.diff(times[valid])>.15))+1
groups=[part[:2*((len(part)-1)//2)+1] for part in np.split(valid,cuts) if len(part)>=3]
frames=np.concatenate(groups)
test_pairs=[]
offset=0
for part in groups:
 i=np.arange(offset,offset+len(part)-2,2)
 test_pairs.extend(zip(i,i+2))
 offset+=len(part)
test_pairs=np.array(test_pairs)
ts1=times[frames]
world1=np.repeat(np.eye(4)[None],len(ts1),axis=0)
world1[:,:3,:3]=Slerp(gt_t,Rotation.from_quat(gt_pose[:,3:7]))(ts1).as_matrix()
world1[:,:3,3]=np.column_stack([np.interp(ts1,gt_t,gt_pose[:,k]) for k in range(3)])
world1=world1@d['base_to_radar']
delta1=np.linalg.inv(world1[test_pairs[:,0]])@world1[test_pairs[:,1]]
truth1=np.column_stack((delta1[:,:3,3],Rotation.from_matrix(delta1[:,:3,:3]).as_euler('XYZ',degrees=True)))
print('independent lab_run1 frames',len(frames),'pairs',len(test_pairs),'segments',len(groups),'strong negative/positive',int((truth1[:,5]<=-4).sum()),int((truth1[:,5]>=4).sum()),flush=True)
assert (truth1[:,5]<=-4).any() and (truth1[:,5]>=4).any()
cache=holdout/'spectra.npz'
if cache.exists():
 saved=np.load(cache); assert np.array_equal(saved['ids'],frames); a=saved['spectra']
else:
 a=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for n,value in enumerate(pool.map(spectrum,frames),1):
   assert np.isfinite(value).all()
   a.append(value)
   if n%256==0:print('converted',n,flush=True)
 a=np.stack(a)
 np.savez_compressed(cache,ids=frames,spectra=a)
a=interpolate(torch.from_numpy(a[:,None]),size=(64,256),mode='bilinear',align_corners=False)[:,0]
a=((a-fit.mean())/fit.std()).cuda()
tx=a[test_pairs].repeat_interleave(3,dim=1)[:,None]
x=make(long_train)
y=torch.from_numpy(label(long_train)[0]).cuda()
results=[]
fig,axes=plt.subplots(1,3,figsize=(14,5))
for ax,seed in zip(axes,(20261102,20261103,20261104)):
 for arm in ('joint','yaw_warmup'):
  torch.manual_seed(seed)
  m=RadarOnlyMilliEgo().cuda().train()
  opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
  for step in range(1120):
   ix=torch.randint(len(x),(24,),device='cuda')
   p=m(x[ix])[:,0]
   l=(p[:,5]-y[ix,5]).square().mean() if arm=='yaw_warmup' and step<560 else loss(p,y[ix])
   assert torch.isfinite(l)
   opt.zero_grad(set_to_none=True);l.backward();opt.step()
  m.eval()
  with torch.inference_mode(): p=torch.cat([m(b)[:,0].cpu() for b in tx.split(32)]).numpy()
  met,trajs=metrics(p,dict(pairs=test_pairs,world_radar=world1,deltas=delta1,timestamps=ts1),np.arange(len(p)),'XYZ')
  q=truth1[:,5]; err=(p[:,:3]-truth1[:,:3])@d['base_to_radar'][:3,:3].T
  row=dict(seed=seed,arm=arm,yaw_rmse=float(np.sqrt(np.mean((p[:,5]-q)**2))),zero_yaw_rmse=float(np.sqrt(np.mean(q*q))),directions={},forward_rmse=float(np.sqrt(np.mean(err[:,0]**2))),lateral_rmse=float(np.sqrt(np.mean(err[:,1]**2))),metrics=met)
  for name,mask in [('negative',q<=-4),('positive',q>=4)]:
   row['directions'][name]=[int((np.sign(q[mask])==np.sign(p[mask,5])).sum()),int(mask.sum())]
  longest=max(range(len(trajs)),key=lambda i:len(trajs[i][0]))
  for i,(P,G,tm) in enumerate(trajs):
   save_tum(holdout/f'{arm}_{seed}_{i}_prediction.tum',P,tm)
   if arm=='joint': save_tum(holdout/f'reference_{i}.tum',G,tm)
   if i==longest:
    if arm=='joint':ax.plot(G[:,0,3],G[:,1,3],'k--',label='Reference')
    ax.plot(P[:,0,3],P[:,1,3],label=arm)
  print(json.dumps(row),flush=True)
  results.append(row)
 ax.set(title=str(seed),xlabel='x [m]',ylabel='y [m]');ax.axis('equal');ax.legend()
fig.suptitle('ColoRadar lab_run1 | independent route | longest valid segment')
fig.tight_layout();fig.savefig(holdout/'trajectory.png',dpi=160)
(holdout/'results.json').write_text(json.dumps(results,indent=2))

