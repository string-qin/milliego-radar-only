from pathlib import Path
s=Path('runs/adc_heatmap_pilot_v1/coloradar_40_train.py').read_text()
exec(chr(10).join(l for l in s[:s.index('train=[load_route')].splitlines() if 'split.json' not in l))
import tempfile
job=Path(tempfile.mkdtemp(prefix='milliego-turn-',dir='/dev/shm'))
print('RESULT DIRECTORY',job,flush=True)
ck=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
train=[load_route(r,False)[:3] for r in split['train']]
val=[(r.name,load_route(r,True)) for r in split['validation']]
def inp(a,p):
 v=interpolate(a[p].cuda().float(),size=(64,256),mode='bilinear',align_corners=False)
 return ((v-ck['mean'])/ck['std']).repeat_interleave(3,dim=1)[:,None]
sizes=np.array([len(r[1]) for r in train])
offset=np.r_[0,np.cumsum(sizes)]
yaws=np.concatenate([y[:,5].numpy() for a,p,y in train])
old=np.concatenate([np.arange(offset[i],offset[i+1]) for i,r in enumerate(split['train']) if r.name.endswith(('arpg_lab_run0','ec_hallways_run0','outdoors_run0','outdoors_run4'))])
all_ids=np.arange(len(yaws))
def bins(pool,edges):
 q=yaws[pool]
 return [pool[(q>=lo)&(q<hi)] for lo,hi in zip(edges[:-1],edges[1:])]
small=bins(old,[-np.inf,-3,3,np.inf])
wide=bins(all_ids,[-np.inf,-6,-3,3,6,np.inf])
print('training',len(yaws),'bins',[len(a) for a in wide],flush=True)
assert min(map(len,small+wide))>0
def ev(m):
 rows=[]
 m.eval()
 for name,(a,p,y,td) in val:
  with torch.inference_mode():z=torch.cat([m(inp(a,b))[:,0].cpu() for b in np.array_split(p,(len(p)+31)//32)]).numpy()
  q=y.numpy()[:,5]; pred=z[:,5];e=pred-q
  met,tr=metrics(z,td,np.arange(len(p)),'XYZ')
  def h(T):return np.unwrap(np.arctan2(T[:,1,0],T[:,0,0]))
  he=np.concatenate([np.rad2deg(np.unwrap(h(P)-h(G))) for P,G,_ in tr])
  ss=met['segments']; lo=abs(q)<1; k=abs(q)>=4
  rows.append(dict(route=name,yaw=float(np.sqrt(np.mean(e**2))),small=float(np.sqrt(np.mean(e[lo]**2))),turn_rmse=float(np.sqrt(np.mean(e[k]**2))),shortfall=float(np.mean(abs(q[k])-np.sign(q[k])*pred[k])),direction=int((np.sign(q[k])==np.sign(pred[k])).sum()),turns=int(k.sum()),heading=float(np.sqrt(np.mean(he**2))),ate=float(np.sqrt(sum(t['poses']*t['ate_initial_rmse_m']**2 for t in ss)/sum(t['poses'] for t in ss)))))
 return rows
m=RadarOnlyMilliEgo().cuda();m.load_state_dict(ck['model'])
results=[dict(arm='baseline',step=0,routes=ev(m))]
for seed in (20261102,20261103,20261104):
 for arm in ('uniform40','turn4','turn40','angle_bins40'):
  torch.manual_seed(seed);rng=np.random.default_rng(seed)
  m.load_state_dict(ck['model'])
  opt=torch.optim.RMSprop(m.parameters(),lr=1e-5,alpha=.9,eps=1e-8)
  for step in range(1,1201):
   m.train()
   if arm=='uniform40': ids=rng.integers(len(yaws),size=24)
   elif arm in ('turn4','turn40'):
    pools=small if arm=='turn4' else bins(all_ids,[-np.inf,-3,3,np.inf])
    ids=np.concatenate([rng.choice(k,n) for k,n in zip(pools,(6,12,6))])
   else: ids=np.concatenate([rng.choice(k,n) for k,n in zip(wide,(3,3,12,3,3))])
   rid=np.searchsorted(offset[1:],ids,side='right'); xs=[];ys=[]
   for i in np.unique(rid):
    a,p,y=train[i]; j=ids[rid==i]-offset[i];xs.append(inp(a,p[j]));ys.append(y[j])
   l=loss(m(torch.cat(xs))[:,0],torch.cat(ys).cuda())
   assert torch.isfinite(l)
   opt.zero_grad(set_to_none=True);l.backward();opt.step()
   if step in (400,1200):
    rows=ev(m)
    rec=dict(seed=seed,arm=arm,step=step,routes=rows)
    results.append(rec)
    (job/'results.json').write_text(json.dumps(results,indent=2))
    print('RESULT',json.dumps(rec),flush=True)
  torch.save(dict(model=m.state_dict(),mean=ck['mean'],std=ck['std']),job/f'{arm}_{seed}.pt')
print('COMPLETE',flush=True)

