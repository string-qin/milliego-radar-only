from pathlib import Path
src=Path('runs/adc_heatmap_pilot_v1/expanded_pilot.py').read_text()
exec(src[:src.index('results={}')])
from scipy.stats import spearmanr
img=imgs['adc']
fit=img[np.unique(pairs[:nt])]
x=((img-fit.mean())/fit.std()).cuda()[pairs].repeat_interleave(3,dim=1)[:,None]
yaw=labels[:,5]
groups=[torch.where(yaw[:nt]<-2)[0],torch.where(yaw[:nt].abs()<=2)[0],torch.where(yaw[:nt]>2)[0]]
print('train counts negative neutral positive',*[len(g) for g in groups],flush=True)
def evaluate(m):
 m.eval()
 with torch.no_grad(): p=torch.cat([m(b)[:,0,5] for b in x.split(32)])
 result={}
 for name,sl in [('train',slice(0,nt)),('val',slice(nt,nt+nv)),('test',slice(nt+nv,None))]:
  a=yaw[sl]; b=p[sl]
  result[name]={'rmse':float((a-b).square().mean().sqrt()),'zero_rmse':float(a.square().mean().sqrt()),'std':float(b.std()),'rho':float(spearmanr(a.cpu(),b.cpu()).statistic),'turns':[]}
  for k in (a<=-2,a>=2):
   result[name]['turns'].append({'n':int(k.sum()),'correct':int((a[k].sign()==b[k].sign()).sum()),'prediction_median':float(b[k].median()) if k.any() else None})
 return result
results={}
for arm in ('uniform','balanced'):
 torch.manual_seed(20261102)
 m=RadarOnlyMilliEgo().cuda()
 opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
 for epoch in range(1,41):
  m.train()
  for _ in range(28):
   if arm=='uniform': ids=torch.randint(nt,(24,),device='cuda')
   else: ids=torch.cat([g[torch.randint(len(g),(8,),device='cuda')] for g in groups])
   p=m(x[ids])[:,0,5]
   loss=(p-yaw[ids]).square().mean()
   assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if epoch in (10,20,40):
   r=evaluate(m); print(arm,epoch,json.dumps(r),flush=True)
   results[f'{arm}_{epoch}']=r
(out/'yaw_learning_diagnostic.json').write_text(json.dumps(results,indent=2))

