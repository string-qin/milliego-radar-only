from pathlib import Path
s=Path('runs/adc_heatmap_pilot_v1/coloradar_40_train.py').read_text()
exec(s[:s.index('train=[load_route')])
ck=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
target=out/'multistep_rotation_pilot_v1'
target.mkdir(exist_ok=True)
names=('arpg_lab_run0','ec_hallways_run0','outdoors_run0','outdoors_run4')
train=[]
for suffix in names:
 r=next(r for r in split['train'] if r.name.endswith(suffix))
 a,p,y,td=load_route(r,False)
 lookup={int(start):i for i,(start,end) in enumerate(p)}
 windows=[]
 for i in range(len(p)):
  w=[i]
  for _ in range(3):
   j=lookup.get(int(p[w[-1],1]))
   if j is None:break
   w.append(j)
  if len(w)==4:windows.append(w)
 train.append((a,p,y,np.array(windows)))
val=[(r.name,load_route(r,True)) for r in split['validation']]
def inp(a,p):
 v=interpolate(a[p].cuda().float(),size=(64,256),mode='bilinear',align_corners=False)
 return ((v-ck['mean'])/ck['std']).repeat_interleave(3,dim=1)[:,None]
def rot(x):
 a,b,c=x.deg2rad().unbind(-1)
 A,B,C=a.cos(),b.cos(),c.cos()
 u,v,w=a.sin(),b.sin(),c.sin()
 return torch.stack([B*C,-B*w,v,A*w+u*v*C,A*C-u*v*w,-u*B,u*w-A*v*C,u*C+A*v*w,A*B],-1).reshape(*x.shape[:-1],3,3)
check=torch.tensor([[25.,-13.,52.]])
np.testing.assert_allclose(rot(check).numpy(),Rotation.from_euler('XYZ',check.numpy(),degrees=True).as_matrix(),atol=1e-6)
def compose(x):
 r=rot(x)
 return r[:,0]@r[:,1]@r[:,2]@r[:,3]
def eval_model(m):
 m.eval();rows=[]
 for name,(a,p,y,td) in val:
  with torch.inference_mode():z=torch.cat([m(inp(a,b))[:,0].cpu() for b in np.array_split(p,(len(p)+31)//32)]).numpy()
  q=y.numpy()[:,5]; e=z[:,5]-q; low=abs(q)<1; big=abs(q)>=4
  met,tr=metrics(z,td,np.arange(len(p)),'XYZ')
  ss=met['segments']
  def h(T):return np.unwrap(np.arctan2(T[:,1,0],T[:,0,0]))
  he=np.concatenate([np.rad2deg(np.unwrap(h(P)-h(G))) for P,G,_ in tr])
  rows.append(dict(route=name,yaw=float(np.sqrt(np.mean(e**2))),low=float(np.sqrt(np.mean(e[low]**2))),short=float(np.mean(abs(q[big])-np.sign(q[big])*z[big,5])),heading=float(np.sqrt(np.mean(he**2))),ate=float(np.sqrt(sum(s['poses']*s['ate_initial_rmse_m']**2 for s in ss)/sum(s['poses'] for s in ss)))))
 return rows
m=RadarOnlyMilliEgo().cuda();m.load_state_dict(ck['model'])
results={'before':eval_model(m)}
print('before',json.dumps(results['before']),flush=True)
for arm in ('single','multistep'):
 torch.manual_seed(20261102)
 rng=np.random.default_rng(20261102)
 m.load_state_dict(ck['model'])
 opt=torch.optim.RMSprop(m.parameters(),lr=1e-5,alpha=.9,eps=1e-8)
 m.train()
 for step in range(400):
  a,p,y,w=train[step%len(train)]
  ix=w[rng.integers(len(w),size=6)]
  z=m(inp(a,p[ix.ravel()]))[:,0];q=y[ix.ravel()].cuda()
  l=loss(z,q)
  if arm=='multistep':
   R=compose(z[:,3:].reshape(-1,4,3));G=compose(q[:,3:].reshape(-1,4,3))
   l=l+.001*((R-G).square().sum((-2,-1)).mean())*(180/np.pi)**2/(2*4*3)
  assert torch.isfinite(l)
  opt.zero_grad(set_to_none=True);l.backward();opt.step()
 results[arm]=eval_model(m)
 torch.save(dict(model=m.state_dict(),mean=ck['mean'],std=ck['std']),target/f'{arm}.pt')
 (target/'results.json').write_text(json.dumps(results,indent=2))
 print(arm,json.dumps(results[arm]),flush=True)

