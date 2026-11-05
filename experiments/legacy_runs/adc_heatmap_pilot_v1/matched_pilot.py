import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.nn.functional import interpolate
from scripts.validate_expanded_training import BASE, ROOT, RadarOnlyMilliEgo, loss
from scripts.validate_projection_diagnosis import metrics

out = ROOT / 'runs/adc_heatmap_pilot_v1'
repo = out / 'coloradar'
sys.path.insert(0, str(repo))
from core.radar import SCRadar, rdsp

from core.calibration import SCRadarCalibration
src = Path('/storage/data/metaiot_data/public_dataset/coloradar')
seq = src / '12_21_2020_arpg_lab_run0'
adc = seq / 'single_chip/adc_samples'
cfg = json.loads((repo / 'dataset/dataset.json').read_text())['calibration']['scradar']
cal = SCRadarCalibration({k: str(src / v) for k, v in cfg.items()})
w = cal.waveform
shape = (w.num_tx, w.num_rx, w.num_chirps_per_frame, w.num_adc_samples_per_chirp, 2)
window = np.blackman(shape[3])
coupling = cal.get_coupling_calibration()

def spectrum(index):
    radar = SCRadar.__new__(SCRadar)
    radar.sensor, radar.calibration = 'scradar', cal
    raw = np.fromfile(adc / 'data' / f'frame_{index}.bin', dtype='<i2').reshape(shape)
    x = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)
    x -= x.mean()
    x *= window
    _, _, nc, ns = radar._get_fft_size(None, None, *x.shape[-2:])
    r = np.fft.fft(x, ns, -1) - coupling
    v = np.fft.fftshift(np.fft.fft(r, nc, -2), -2) * rdsp.velocity_compensation(shape[0], nc)
    # Parseval preserves the elevation-summed power without constructing its FFT cube.
    layout = rdsp.virtual_array(v, cal.antenna.txl, cal.antenna.rxl)
    ne, na, _, _ = radar._get_fft_size(*layout.shape)
    az = np.fft.fftshift(np.fft.fft(layout, na, 1), 1)
    power = (np.abs(az)**2).sum(axis=(0, 2)) * ne
    return (10 * np.log10(1 + power / max(np.quantile(power, .3), 1e-12))).astype(np.float32)

t = np.loadtxt(adc / 'timestamps.txt')
d = dict(np.load(BASE / 'labels.npz'))
pairs = d['pairs']
j = np.searchsorted(t, d['timestamps']).clip(1, len(t)-1)
nearest = np.where(abs(t[j]-d['timestamps']) < abs(t[j-1]-d['timestamps']), j, j-1)
gt_t = np.loadtxt(seq/'groundtruth/timestamps.txt')
gt_pose = np.loadtxt(seq/'groundtruth/groundtruth_poses.txt')
match = abs(t[nearest]-d['timestamps']) < .012
valid = match[pairs].all(axis=1) & (np.diff(nearest[pairs],axis=1)[:,0]==1)
valid &= ((t[nearest[pairs]]>=gt_t[0]) & (t[nearest[pairs]]<=gt_t[-1])).all(axis=1)
tr = np.flatnonzero(valid & (d['split']==0))[::5]
va = np.flatnonzero(valid & (d['split']==1))[::5]
test = np.flatnonzero(valid & (d['split']==2))
groups = np.split(test, np.flatnonzero(pairs[test[1:],0]!=pairs[test[:-1],1])+1)
test = max(groups,key=len)
start=(len(test)-64)//2
assert start>=0
test=test[start:start+64]
rows=np.r_[tr,va,test]
p=pairs[rows]
frames=np.unique(p)
adc_ids=nearest[frames]
cache=out/'matched_spectra.npz'
if cache.exists():
    saved=np.load(cache); assert np.array_equal(saved['adc_ids'],adc_ids); spectra=saved['spectra']
else:
    saved=np.load(out/'lab_run0_adc_smoke.npz')
    np.testing.assert_allclose(spectrum(int(saved['adc_ids'][0])),saved['spectra'][0],rtol=2e-5,atol=2e-5)
    print('verified accelerated spectrum against upstream; frames',len(frames),'pairs',list(map(len,(tr,va,test))),flush=True)
    values=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,img in enumerate(pool.map(spectrum,adc_ids),1):
            assert np.isfinite(img).all()
            values.append(img)
            if n%32==0:print('converted',n,'of',len(frames),flush=True)
    spectra=np.stack(values)
    np.savez_compressed(cache,adc_ids=adc_ids,spectra=spectra)
torch.set_num_threads(3)
ts=t[adc_ids]
world=np.repeat(np.eye(4)[None],len(ts),axis=0)
world[:,:3,:3]=Slerp(gt_t,Rotation.from_quat(gt_pose[:,3:7]))(ts).as_matrix()
world[:,:3,3]=np.column_stack([np.interp(ts,gt_t,gt_pose[:,k]) for k in range(3)])
world=world@d['base_to_radar']
pairs=np.searchsorted(frames,p)
delta=np.linalg.inv(world[pairs[:,0]])@world[pairs[:,1]]
labels=np.column_stack((delta[:,:3,3],Rotation.from_matrix(delta[:,:3,:3]).as_euler('XYZ',degrees=True)))
labels=torch.from_numpy(labels.astype(np.float32)).cuda()
imgs={'adc':interpolate(torch.from_numpy(spectra[:,None]),size=(64,256),mode='bilinear',align_corners=False)[:,0],
      'pointcloud':torch.from_numpy(np.load(BASE/'images_raw.npy')[frames]).float()}
nt,nv=len(tr),len(va)
results={}
for arm,img in imgs.items():
    torch.manual_seed(20261102)
    fit=img[np.unique(pairs[:nt])]
    img=((img-fit.mean())/fit.std().clamp_min(1e-6)).cuda()
    x=img[pairs].repeat_interleave(3,dim=1)[:,None]
    m=RadarOnlyMilliEgo().cuda()
    opt=torch.optim.RMSprop(m.parameters(),lr=1e-4,alpha=.9,eps=1e-8)
    best=float('inf')
    for epoch in range(1,21):
        m.train()
        for ids in torch.randperm(nt,device='cuda').split(16):
            opt.zero_grad(set_to_none=True)
            l=loss(m(x[ids])[:,0],labels[ids])
            assert torch.isfinite(l)
            l.backward();opt.step()
        m.eval()
        with torch.inference_mode():
            pred=m(x[nt:nt+nv])[:,0]
            score=float(loss(pred,labels[nt:nt+nv]))
        if score<best:
            best=score
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            selected=epoch
    m.load_state_dict(state)
    with torch.inference_mode():
        pred=torch.cat([m(b)[:,0].cpu() for b in x[nt+nv:].split(16)]).numpy()
    dd={'pairs':pairs,'world_radar':world,'deltas':delta,'timestamps':ts}
    met,_=metrics(pred,dd,np.arange(nt+nv,len(pairs)),'XYZ')
    r=Rotation.from_euler('XYZ',pred[:,3:],degrees=True)
    gt=Rotation.from_matrix(delta[nt+nv:,:3,:3])
    mask=np.rad2deg(gt.magnitude())>=4
    results[arm]={'epoch':selected,'validation_loss':best,'metrics':met,'turn_count':int(mask.sum()),
       'turn_rmse_deg':float(np.sqrt(np.mean(np.rad2deg((gt[mask].inv()*r[mask]).magnitude())**2))) if mask.any() else None}
    print(arm,json.dumps(results[arm]),flush=True)
(out/'matched_pilot_results.json').write_text(json.dumps({'split_counts':[nt,nv,len(test)],'seed':20261102,'epochs':20,'results':results},indent=2))

