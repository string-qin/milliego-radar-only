from pathlib import Path
s=Path('runs/adc_heatmap_pilot_v1/multisequence_gap2.py').read_text()
pre=s[:s.index('routes=')]
exec(pre)
pilot=out.parent
out=pilot/'lab_interval_sweep_v1'
out.mkdir(exist_ok=True)
base_img=img
def track(n):
 r=seq.with_name(f'12_21_2020_arpg_lab_run{n}')
 t=np.loadtxt(r/'single_chip/adc_samples/timestamps.txt')
 g=np.loadtxt(r/'groundtruth/timestamps.txt')
 p=np.loadtxt(r/'groundtruth/groundtruth_poses.txt')
 path=pilot/'lab_run1_gap2_holdout/spectra.npz' if n==1 else pilot/'multisequence_gap2_v1'/f'lab_run{n}_spectra.npz'
 with np.load(path) as f:
  ids=f['ids' if n==1 else 'frames']; a=f['spectra']
 ts=t[ids]
 W=np.repeat(np.eye(4)[None],len(ts),axis=0)
 W[:,:3,:3]=Slerp(g,Rotation.from_quat(p[:,3:7]))(ts).as_matrix()
 W[:,:3,3]=np.column_stack([np.interp(ts,g,p[:,k]) for k in range(3)])
 W=W@d['base_to_radar']
 X=interpolate(torch.from_numpy(a[:,None]),size=(64,256),mode='bilinear',align_corners=False)[:,0]
 X=((X-fit.mean())/fit.std()).cuda()
 groups=np.split(np.arange(len(ids)),np.flatnonzero((np.diff(ids)!=1)|(np.diff(ts)>.15))+1)
 return X,W,ts,groups
T={n:track(n) for n in (1,2,4)}
def pair_groups(groups,gap):
 return np.concatenate([np.column_stack([g[:-gap],g[gap:]]) for g in groups if len(g)>gap])
def labels_for(W,p):
 D=np.linalg.inv(W[p[:,0]])@W[p[:,1]]
 return np.column_stack([D[:,:3,3],Rotation.from_matrix(D[:,:3,:3]).as_euler('XYZ',degrees=True)]).astype(np.float32),D
def inp(X,p):
 return X[p].repeat_interleave(3,dim=1)[:,None]
# Same starting frames for all three training arms.
p0=pairs[:nt]
i0=np.flatnonzero((p0[:-2,1]==p0[1:-1,0])&(p0[1:-1,1]==p0[2:,0]))
X2,W2,_,G2=T[2]
results=[]
for gap in (1,2,3):
 p0g=np.column_stack([p0[i0,0],p0[i0+gap-1,1]])
 p2=np.concatenate([np.column_stack([a[:-3],a[gap:len(a)-3+gap]]) for a in G2 if len(a)>3])
 train_X=torch.cat([base_img,X2])
 train_p=torch.from_numpy(np.concatenate([p0g,p2+len(base_img)])).cuda()
 train_y=torch.from_numpy(np.concatenate([labels_for(world,p0g)[0],labels_for(W2,p2)[0]])).cuda()
 torch.manual_seed(20261102)
 m=RadarOnlyMilliEgo().cuda().train()
 opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
 for step in range(1120):
  ix=torch.randint(len(train_p),(24,),device='cuda')
  p=m(inp(train_X,train_p[ix]))[:,0]; y=train_y[ix]
  l=(p[:,5]-y[:,5]).square().mean() if step<560 else loss(p,y)
  assert torch.isfinite(l)
  opt.zero_grad(set_to_none=True);l.backward();opt.step()
 m.eval()
 for route in (1,4):
  X,W,ts,groups=T[route]
  groups=[a[:((len(a)-1)//6)*6+1] for a in groups if len(a)>=7]
  short=np.concatenate([np.column_stack([a[:-gap:gap],a[gap::gap]]) for a in groups])
  common=np.concatenate([np.column_stack([a[:-6:6],a[6::6]]) for a in groups])
  with torch.inference_mode():p=torch.cat([m(inp(X,b))[:,0].cpu() for b in np.array_split(short,(len(short)+31)//32)]).numpy()
  P=np.repeat(np.eye(4)[None],len(p),axis=0)
  P[:,:3,:3]=Rotation.from_euler('XYZ',p[:,3:],degrees=True).as_matrix()
  P[:,:3,3]=p[:,:3]
  P=P.reshape(len(common),6//gap,4,4)
  C=P[:,0]
  for j in range(1,P.shape[1]):C=C@P[:,j]
  pred=np.column_stack([C[:,:3,3],Rotation.from_matrix(C[:,:3,:3]).as_euler('XYZ',degrees=True)])
  y,D=labels_for(W,common)
  met,trajs=metrics(pred,dict(pairs=common,world_radar=W,deltas=D,timestamps=ts),np.arange(len(common)),'XYZ')
  seg=met['segments']; yaw=y[:,5]; k=abs(yaw)>=4
  row=dict(gap=gap,route=route,train_pairs=len(train_p),test_pairs=len(common),ate=float(np.sqrt(sum(z['poses']*z['ate_initial_rmse_m']**2 for z in seg)/sum(z['poses'] for z in seg))),path=sum(z['path_length_m'] for z in seg),yaw=float(np.sqrt(np.mean((pred[:,5]-yaw)**2))),turn_correct=int((np.sign(pred[k,5])==np.sign(yaw[k])).sum()),turns=int(k.sum()),translation=met['pair_translation_rmse_m'])
  results.append(row)
  print(json.dumps(row),flush=True)
  (out/'results.json').write_text(json.dumps(results,indent=2))
  for j,(P,G,t) in enumerate(trajs):
   save_tum(out/f'run{route}_gap{gap}_seg{j}.tum',P,t)
   if gap==1:save_tum(out/f'run{route}_reference_seg{j}.tum',G,t)

