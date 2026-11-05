from pathlib import Path
source=Path('runs/adc_heatmap_pilot_v1/multisequence_gap2.py').read_text()
prefix=source[:source.index('routes=')]
prefix=prefix.replace("out=out/'multisequence_gap2_v1'","out=out/'coloradar_route_split_v1'")
prefix=prefix.replace("def load_route(number):","def load_route(route, testing):").replace(" route=seq.with_name(f'12_21_2020_arpg_lab_run{number}')"," number=route.name")
prefix=prefix.replace("number!=4","not testing").replace("f'lab_run{number}_spectra.npz'","f'{number}_spectra.npz'")
prefix=prefix.replace(" A=interpolate(torch.from_numpy(a[:,None]),size=(64,256),mode='bilinear',align_corners=False)[:,0]\n A=((A-fit.mean())/fit.std()).cuda()"," A=torch.from_numpy(a.astype(np.float16))").replace("torch.from_numpy(Y.astype(np.float32)).cuda()","torch.from_numpy(Y.astype(np.float32))")
exec(compile(prefix,'route_preprocessing','exec'))
from collections import defaultdict
from time import monotonic
all_routes=defaultdict(list)
for route in sorted(src.glob('*_run*')):
 if (route/'single_chip/adc_samples/timestamps.txt').is_file() and (route/'groundtruth/groundtruth_poses.txt').is_file():
  all_routes[route.name.split('_',3)[-1].rsplit('_run',1)[0]].append(route)
assert sum(map(len,all_routes.values()))==52,[(k,len(v)) for k,v in all_routes.items()]
test_indices={'arpg_lab':(4,),'ec_hallways':(4,),'outdoors':(8,9),'aspen':(11,), 'longboard':(7,), 'edgar_classroom':(5,), 'edgar_army':(5,)}
val_indices={'ec_hallways':(3,),'outdoors':(7,),'aspen':(10,),'edgar_classroom':(4,)}
split={k:[] for k in ('train','validation','test')}
for scene,routes in all_routes.items():
 for route in routes:
  i=int(route.name.rsplit('_run',1)[1])
  kind='test' if i in test_indices[scene] else 'validation' if i in val_indices.get(scene,()) else 'train'
  split[kind].append(route)
assert list(map(len,split.values()))==[40,4,8]
(out/'split.json').write_text(json.dumps({k:[s.name for s in v] for k,v in split.items()},indent=2))
print('split', {k:len(v) for k,v in split.items()},'outdoor', {k:[s.name for s in v if 'outdoors' in s.name] for k,v in split.items()},flush=True)
train=[load_route(route,False)[:3] for route in split['train']]
validation=[load_route(route,True) for route in split['validation']]
lengths=np.array([len(r[1]) for r in train]); offsets=np.r_[0,np.cumsum(lengths)]
mean=float(fit.mean()); std=float(fit.std())
def input_tensor(a,p):
 v=a[p].to(device='cuda',dtype=torch.float32)
 v=interpolate(v,size=(64,256),mode='bilinear',align_corners=False)
 return ((v-mean)/std).repeat_interleave(3,dim=1)[:,None]
def evaluate(m,data):
 a,p,y,td=data
 m.eval()
 with torch.inference_mode():
  pred=torch.cat([m(input_tensor(a,b))[:,0].cpu() for b in np.array_split(p,(len(p)+31)//32)])
 score=float(loss(pred,y))
 return score,pred.numpy()
print('TRAIN START','routes',len(train),'pairs',int(offsets[-1]),flush=True)
seed=20261102
torch.manual_seed(seed)
rng=np.random.default_rng(seed)
m=RadarOnlyMilliEgo().cuda()
opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
best=float('inf')
history=[]
for step in range(1,8961):
 m.train()
 ids=rng.integers(offsets[-1],size=24)
 route_ids=np.searchsorted(offsets[1:],ids,side='right')
 xs=[]; ys=[]
 for route_id in np.unique(route_ids):
  a,p,y=train[route_id]
  ix=ids[route_ids==route_id]-offsets[route_id]
  xs.append(input_tensor(a,p[ix]));ys.append(y[ix])
 target=torch.cat(ys).cuda()
 pred=m(torch.cat(xs))[:,0]
 l=(pred[:,5]-target[:,5]).square().mean() if step<=1120 else loss(pred,target)
 assert torch.isfinite(l)
 opt.zero_grad(set_to_none=True);l.backward();opt.step()
 if step in (1120,2240,4480,6720,8960):
  scores=[evaluate(m,v)[0] for v in validation]
  score=float(np.mean(scores))
  history.append(dict(step=step,score=score,route_scores=scores))
  (out/'validation.json').write_text(json.dumps(history,indent=2))
  print('VALIDATION',step,score,flush=True)
  if step>1120 and score<best:
   best=score
   torch.save(dict(model=m.state_dict(),step=step,mean=mean,std=std),out/'best.pt')
ck=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
m.load_state_dict(ck['model'])
results=[]
for route in split['test']:
 data=load_route(route,True)
 a,p,y,td=data
 _,pred=evaluate(m,data)
 met,trajs=metrics(pred,td,np.arange(len(p)),'XYZ')
 e=(pred[:,:3]-y.numpy()[:,:3])@d['base_to_radar'][:3,:3].T
 yaw=y.numpy()[:,5]; mask=abs(yaw)>=4
 seg=met['segments']
 row=dict(route=route.name,step=ck['step'],pairs=len(p),ate=float(np.sqrt(sum(s['poses']*s['ate_initial_rmse_m']**2 for s in seg)/sum(s['poses'] for s in seg))),path=sum(s['path_length_m'] for s in seg),yaw_rmse=float(np.sqrt(np.mean((pred[:,5]-yaw)**2))),turn_correct=int((np.sign(pred[mask,5])==np.sign(yaw[mask])).sum()),turns=int(mask.sum()),forward_rmse=float(np.sqrt(np.mean(e[:,0]**2))),lateral_rmse=float(np.sqrt(np.mean(e[:,1]**2))))
 for i,(P,G,times) in enumerate(trajs):
  save_tum(out/f'{route.name}_{i}_prediction.tum',P,times)
  save_tum(out/f'{route.name}_{i}_reference.tum',G,times)
 results.append(row)
 (out/'results.json').write_text(json.dumps(results,indent=2))
 print('TEST',json.dumps(row),flush=True)

