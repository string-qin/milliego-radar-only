from pathlib import Path
src=Path('runs/adc_heatmap_pilot_v1/expanded_pilot.py').read_text()
exec(src[:src.index('results={}')])
img=imgs['adc']; fit=img[np.unique(pairs[:nt])]
x=((img-fit.mean())/fit.std()).cuda()[pairs].repeat_interleave(3,dim=1)[:,None]
y=labels
results=[]
for seed in (20261102,20261103,20261104):
 for arm,weight in [('original',1),('yaw10',10)]:
  torch.manual_seed(seed)
  m=RadarOnlyMilliEgo().cuda()
  opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
  m.train()
  for step in range(1120):
   ids=torch.randint(nt,(24,),device='cuda')
   p=m(x[ids])[:,0]; e=p-y[ids]
   l=loss(p,y[ids])+((weight-1)*.001/3)*e[:,5].square().mean()
   assert torch.isfinite(l)
   opt.zero_grad(set_to_none=True); l.backward();opt.step()
  m.eval()
  with torch.inference_mode(): p=torch.cat([m(b)[:,0].cpu() for b in x.split(32)]).numpy()
  target=y.cpu().numpy()
  row={'seed':seed,'arm':arm}
  for name,idx in [('val',np.arange(nt,nt+nv)),('test',np.arange(nt+nv,len(pairs)))]:
   a=target[idx,5]; b=p[idx,5]; k=abs(a)>=2
   row[name]={'yaw_rmse_deg':float(np.sqrt(np.mean((a-b)**2))),'turn_correct':int((np.sign(a[k])==np.sign(b[k])).sum()),'turn_n':int(k.sum()),'translation_rmse_m':float(np.sqrt(np.mean(np.sum((p[idx,:3]-target[idx,:3])**2,axis=1))))}
   if name=='test':
    met,_=metrics(p[idx],{'pairs':pairs,'world_radar':world,'deltas':delta,'timestamps':ts},idx,'XYZ')
    row[name]['ate_m']=met['ate_initial_rmse_m']
  print(json.dumps(row),flush=True)
  results.append(row)
(out/'joint_yaw_weight_diagnostic.json').write_text(json.dumps({'steps':1120,'batch':24,'results':results},indent=2))

