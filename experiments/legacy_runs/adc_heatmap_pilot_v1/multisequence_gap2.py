from pathlib import Path
g=Path('runs/adc_heatmap_pilot_v1/gap2_diagnostic.py').read_text()
exec(g[:g.index('results=[]')])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.validate_projection_diagnosis import save_tum
out=out/'multisequence_gap2_v1'
out.mkdir(exist_ok=True)

def load_route(number):
 global adc
 route=seq.with_name(f'12_21_2020_arpg_lab_run{number}')
 adc=route/'single_chip/adc_samples'
 times=np.loadtxt(adc/'timestamps.txt')
 gt_t=np.loadtxt(route/'groundtruth/timestamps.txt')
 pose=np.loadtxt(route/'groundtruth/groundtruth_poses.txt')
 j=np.searchsorted(gt_t,times).clip(1,len(gt_t)-1)
 ids=np.flatnonzero((times>=gt_t[0])&(times<=gt_t[-1])&(gt_t[j]-gt_t[j-1]<=.2))
 cuts=np.flatnonzero((np.diff(ids)!=1)|(np.diff(times[ids])>.15))+1
 groups=[a for a in np.split(ids,cuts) if len(a)>=3]
 frames=np.concatenate(groups)
 pair=np.concatenate([np.column_stack((np.arange(start,start+len(a)-2,1 if number!=4 else 2),np.arange(start+2,start+len(a),1 if number!=4 else 2))) for a,start in zip(groups,np.r_[0,np.cumsum(list(map(len,groups)))[:-1]])])
 times=times[frames]
 W=np.repeat(np.eye(4)[None],len(frames),axis=0)
 W[:,:3,:3]=Slerp(gt_t,Rotation.from_quat(pose[:,3:7]))(times).as_matrix()
 W[:,:3,3]=np.column_stack([np.interp(times,gt_t,pose[:,i]) for i in range(3)])
 W=W@d['base_to_radar']
 D=np.linalg.inv(W[pair[:,0]])@W[pair[:,1]]
 Y=np.column_stack((D[:,:3,3],Rotation.from_matrix(D[:,:3,:3]).as_euler('XYZ',degrees=True)))
 cache=out/f'lab_run{number}_spectra.npz'
 if cache.exists():
  saved=np.load(cache);assert np.array_equal(saved['frames'],frames);a=saved['spectra']
 else:
  acc=[]
  with ThreadPoolExecutor(max_workers=8) as pool:
   for n,value in enumerate(pool.map(spectrum,frames),1):
    assert np.isfinite(value).all();acc.append(value)
    if n%256==0:print('converted',number,n,len(frames),flush=True)
  a=np.stack(acc)
  np.savez_compressed(cache,frames=frames,spectra=a)
 A=interpolate(torch.from_numpy(a[:,None]),size=(64,256),mode='bilinear',align_corners=False)[:,0]
 A=((A-fit.mean())/fit.std()).cuda()
 print('route',number,'pairs',len(pair),flush=True)
 return A,pair,torch.from_numpy(Y.astype(np.float32)).cuda(),dict(pairs=pair,world_radar=W,deltas=D,timestamps=times)
routes=[(img,long_train,torch.from_numpy(label(long_train)[0]).cuda())]
for n in (2,3):routes.append(load_route(n)[:3])
A,P,Y,td=load_route(4)
truth=Y.cpu().numpy()
counts=np.cumsum([0]+[len(r[1]) for r in routes])
train_images=torch.cat([r[0] for r in routes])
train_pairs=np.concatenate([r[1]+offset for r,offset in zip(routes,np.r_[0,np.cumsum([len(r[0]) for r in routes])[:-1]])])
train_pairs=torch.from_numpy(train_pairs).cuda()
train_y=torch.cat([r[2] for r in routes])
fig,axes=plt.subplots(1,3,figsize=(14,5))
results=[]
for seed,ax in zip((20261102,20261103,20261104),axes):
 for arm,n in [('single',counts[1]),('multi',counts[-1])]:
  torch.manual_seed(seed)
  m=RadarOnlyMilliEgo().cuda().train()
  opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
  for step in range(2240):
   ix=torch.randint(int(n),(24,),device='cuda')
   inp=train_images[train_pairs[ix]].repeat_interleave(3,dim=1)[:,None]
   p=m(inp)[:,0]; y=train_y[ix]
   l=(p[:,5]-y[:,5]).square().mean() if step<1120 else loss(p,y)
   assert torch.isfinite(l)
   opt.zero_grad(set_to_none=True);l.backward();opt.step()
  m.eval()
  with torch.inference_mode():p=torch.cat([m(A[b].repeat_interleave(3,dim=1)[:,None])[:,0].cpu() for b in np.array_split(P,(len(P)+31)//32)]).numpy()
  met,trajs=metrics(p,td,np.arange(len(p)),'XYZ')
  err=(p[:,:3]-truth[:,:3])@d['base_to_radar'][:3,:3].T
  q=truth[:,5];k=abs(q)>=4
  s=met['segments']
  ate=float(np.sqrt(sum(z['poses']*z['ate_initial_rmse_m']**2 for z in s)/sum(z['poses'] for z in s)))
  row=dict(seed=seed,arm=arm,train_pairs=int(n),ate=ate,yaw=float(np.sqrt(np.mean((p[:,5]-q)**2))),zero=float(np.sqrt(np.mean(q*q))),correct=int((np.sign(q[k])==np.sign(p[k,5])).sum()),turns=int(k.sum()),forward=float(np.sqrt(np.mean(err[:,0]**2))),lateral=float(np.sqrt(np.mean(err[:,1]**2))),path=sum(z['path_length_m'] for z in s),segments=s)
  results.append(row)
  (out/'results.json').write_text(json.dumps(results,indent=2))
  torch.save({'model':m.state_dict(),'mean':float(fit.mean()),'std':float(fit.std()),'steps':2240},out/f'{arm}_{seed}.pt')
  for i,(pred,ref,tm) in enumerate(trajs):
   save_tum(out/f'{arm}_{seed}_{i}.tum',pred,tm)
   if arm=='single':save_tum(out/f'reference_{i}.tum',ref,tm)
  pred,ref,_=max(trajs,key=lambda r:len(r[0]))
  if arm=='single':ax.plot(ref[:,0,3],ref[:,1,3],'k--',label='Reference')
  ax.plot(pred[:,0,3],pred[:,1,3],label=arm)
  print(json.dumps({k:v for k,v in row.items() if k!='segments'}),flush=True)
 ax.set(title=str(seed),xlabel='x [m]',ylabel='y [m]');ax.axis('equal');ax.legend()
fig.suptitle('ColoRadar lab_run4 | radar-only | longest valid segment')
fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=160)

