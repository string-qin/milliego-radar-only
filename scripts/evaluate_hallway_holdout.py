import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np, torch
from scipy.spatial.transform import Rotation,Slerp
from scripts.prepare_single_sequence import project,SRC
from scripts.validate_projection_diagnosis import metrics
from milliego.model import RadarOnlyMilliEgo
from milliego.model_no_attention import RadarOnlyNoAttention
root=Path.cwd()
out=root/'runs/hallway_holdout_v1'
out.mkdir(exist_ok=True)
arms={'attention':(RadarOnlyMilliEgo,'expanded_training_v1'),'no_attention':(RadarOnlyNoAttention,'no_attention_matched_v1')}
models={}
for arm,(cls,run) in arms.items():
 for seed in (20261102,20261103,20261104):
  cp=torch.load(root/'runs'/run/f'expanded_data_seed{seed}'/'full_200_epochs.pt',map_location='cpu',weights_only=True)
  m=cls().cuda()
  m.load_state_dict(cp['model'])
  m.eval()
  models[arm,seed]=m,cp['normalization_mean']
ext=(SRC/'calib/transforms/base_to_single_chip.txt').read_text().splitlines()
extrinsic=np.eye(4)
extrinsic[:3,3]=np.fromstring(ext[0],sep=' ')
extrinsic[:3,:3]=Rotation.from_quat(np.fromstring(ext[1],sep=' ')).as_matrix()
rows=[]
with torch.inference_mode():
 for seq in sorted(SRC.glob('12_21_2020_ec_hallways_run*')):
  t=np.loadtxt(seq/'single_chip/pointclouds/timestamps.txt')
  gt=np.loadtxt(seq/'groundtruth/timestamps.txt')
  poses=np.loadtxt(seq/'groundtruth/groundtruth_poses.txt')
  j=np.clip(np.searchsorted(gt,t),1,len(gt)-1)
  ids=np.flatnonzero((t>=gt[0])&(t<=gt[-1])&((gt[j]-gt[j-1])<=.2))
  groups=np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1)
  for group in groups:
   n=(len(group)-1)//5
   if n<1:continue
   ids=group[:5*n+1]
   ts=t[ids]
   W=np.repeat(np.eye(4)[None],len(ts),axis=0)
   W[:,:3,:3]=Slerp(gt,Rotation.from_quat(poses[:,3:7]))(ts).as_matrix()
   W[:,:3,3]=np.column_stack([np.interp(ts,gt,poses[:,k]) for k in range(3)])
   W=(W@extrinsic)[::5]
   d={'pairs':np.column_stack((np.arange(n),np.arange(1,n+1))),'world_radar':W,'deltas':np.linalg.inv(W[:-1])@W[1:],'timestamps':ts[::5]}
   raw=np.stack([project(np.fromfile(seq/f'single_chip/pointclouds/data/radar_pointcloud_{i}.bin',dtype='<f4').reshape(-1,5))[0] for i in ids])
   for (arm,seed),(model,mean) in models.items():
    frames=torch.from_numpy(raw).cuda().float()/255-mean
    preds=[]
    for k in range(0,len(ids)-1,32):
     a=frames[k:min(k+32,len(ids)-1)]
     b=frames[k+1:k+1+len(a)]
     inp=torch.cat([a[:,None].expand(-1,3,-1,-1),b[:,None].expand(-1,3,-1,-1)],1)
     preds.append(model(inp[:,None])[:,0].cpu().numpy())
    p=np.concatenate(preds)
    T=np.repeat(np.eye(4)[None],len(p),axis=0)
    T[:,:3,:3]=Rotation.from_euler('XYZ',p[:,3:],degrees=True).as_matrix()
    T[:,:3,3]=p[:,:3]
    T=T.reshape(n,5,4,4)
    D=T[:,0]
    for k in range(1,5):D=D@T[:,k]
    p=np.column_stack([D[:,:3,3],Rotation.from_matrix(D[:,:3,:3]).as_euler('XYZ',degrees=True)])
    m,_=metrics(p,d,np.arange(n),'XYZ')
    rows.append(dict(sequence=seq.name,start=int(ids[0]),end=int(ids[-1]),arm=arm,seed=seed,**m))
   print(seq.name,int(ids[0]),int(ids[-1]),flush=True)
assert len({r['sequence'] for r in rows})==5
(out/'results.json').write_text(json.dumps(rows,indent=2))
for name in sorted({r['sequence'] for r in rows}):
 for arm in arms:
  rr=[r for r in rows if r['sequence']==name and r['arm']==arm]
  w=np.array([r['segments'][0]['poses'] for r in rr])
  ate=np.sqrt(np.average([r['ate_initial_rmse_m']**2 for r in rr],weights=w))
  print(name,arm,round(float(ate),3),len(rr),flush=True)

