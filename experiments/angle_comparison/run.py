"""Recover one-step rotations from saved trajectories; never retrain or calibrate."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/dev/shm/milliego-angle-comparison-20261105'); root.mkdir(exist_ok=True)
scratch=Path('/dev/shm/milliego-author-scratch-20261105')
dirs={
 'yaw_warmup':Path('/home/metaiot_guest/Workspace/qinzx/milliEgo/runs/adc_heatmap_pilot_v1/coloradar_route_split_v1'),
 'frozen_cnn':Path('/dev/shm/milliego-author-spectrum-20261105'),
 'scratch_cnn':scratch,
}
routes=[r['route'] for r in json.loads((scratch/'summary.json').read_text())]
colors={'yaw_warmup':'tab:green','frozen_cnn':'tab:orange','scratch_cnn':'tab:blue'}
rows=[]; route_data={}
fig,axes=plt.subplots(2,4,figsize=(23,10))
def increments(tum):
    R=Rotation.from_quat(tum[:,4:8])
    return R[:-1].inv()*R[1:]
for ax,route in zip(axes.flat,routes):
    gathered={m:[] for m in dirs}; refs=[]; geos={m:[] for m in dirs}
    files=sorted(scratch.glob(route+'*_reference.tum'))
    first_time=min(np.loadtxt(f,ndmin=2)[0,0] for f in files)
    for si,f in enumerate(files):
        g=np.loadtxt(f,ndmin=2); rg=increments(g); yz=rg.as_euler('XYZ',degrees=True)[:,2]
        refs.append(yz); model_values={}
        ax.plot(g[1:,0]-first_time,yz,'k--',lw=1,label='Reference' if si==0 else None)
        for model,directory in dirs.items():
            number=f.name.split('_segment')[1].split('_')[0]
            name=f.name if model!='yaw_warmup' else f'{route}_{number}_reference.tum'
            reference=np.loadtxt(directory/name,ndmin=2)
            assert reference.shape==g.shape and np.allclose(reference[:,0],g[:,0],atol=1e-6,rtol=0)
            assert np.max((increments(reference).inv()*rg).magnitude())<1e-7
            p=np.loadtxt(directory/name.replace('_reference','_prediction'),ndmin=2)
            assert p.shape==g.shape and np.allclose(p[:,0],g[:,0],atol=1e-6,rtol=0)
            rp=increments(p); yp=rp.as_euler('XYZ',degrees=True)[:,2]
            gathered[model].append(yp); model_values[model]=yp
            geos[model].append((rg.inv()*rp).magnitude()*180/np.pi)
            ax.plot(g[1:,0]-first_time,yp,color=colors[model],lw=.8,alpha=.8,label=model if si==0 else None)
        for j in range(len(yz)):
            rows.append(dict(route=route,segment=f.stem,time=float(g[j+1,0]),dt=float(g[j+1,0]-g[j,0]),reference=float(yz[j]),
                             **{m:float(v[j]) for m,v in model_values.items()}))
    route_data[route]=(np.concatenate(refs),{m:np.concatenate(v) for m,v in gathered.items()},
                       {m:np.concatenate(v) for m,v in geos.items()})
    ax.set_title(route,fontsize=9); ax.set_xlabel('Time [s]'); ax.set_ylabel('One-step XYZ z [deg]'); ax.grid(alpha=.25)
axes.flat[0].legend(fontsize=8)
fig.suptitle('Same held-out timestamps | supervised XYZ z angle | no output calibration')
fig.tight_layout(); fig.savefig(root/'angle_time_comparison.png',dpi=150)

def score(y,p,geo):
    error=(p-y+180)%360-180
    masks={'all':np.ones(len(y),dtype=bool),'near_straight_abs_lt1':np.abs(y)<1,
           'small_turn_1to3':(np.abs(y)>=1)&(np.abs(y)<3),'strong_turn_abs_ge3':np.abs(y)>=3,
           'positive_ge3':y>=3,'negative_le_minus3':y<=-3}
    result={}
    for name,mask in masks.items():
        if not mask.any(): continue
        r=dict(n=int(mask.sum()),yaw_rmse_deg=float(np.sqrt(np.mean(error[mask]**2))),
               yaw_mae_deg=float(np.mean(np.abs(error[mask]))),signed_bias_deg=float(error[mask].mean()),
               rotation_geodesic_rmse_deg=float(np.sqrt(np.mean(geo[mask]**2))),
               reference_mean_deg=float(y[mask].mean()),prediction_mean_deg=float(p[mask].mean()))
        if name in ('strong_turn_abs_ge3','positive_ge3','negative_le_minus3'):
            r['direction_accuracy']=float(np.mean(np.sign(p[mask])==np.sign(y[mask])))
            r['signed_magnitude_shortfall_deg']=float(np.mean(np.abs(y[mask])-p[mask]*np.sign(y[mask])))
        result[name]=r
    return result
report={'definition':'Intrinsic XYZ Euler z in radar coordinates, degrees; positive/negative are label signs, not guaranteed physical left/right. Full 3D geodesic errors also provided.',
        'scope':'Same 8 held-out training routes; single checkpoints; multiple training settings differ. No causal attribution to warmup alone.',
        'by_route':{},'pooled':{}}
for route,(y,p,g) in route_data.items(): report['by_route'][route]={m:score(y,p[m],g[m]) for m in dirs}
for m in dirs:
    report['pooled'][m]=score(np.concatenate([d[0] for d in route_data.values()]),
                            np.concatenate([d[1][m] for d in route_data.values()]),np.concatenate([d[2][m] for d in route_data.values()]))
(root/'angles.json').write_text(json.dumps(report,indent=2))
with (root/'pairs.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(13,5))
groups=['near_straight_abs_lt1','small_turn_1to3','strong_turn_abs_ge3']
for i,m in enumerate(dirs):
    axes[0].bar(np.arange(3)+(i-1)*.24,[report['pooled'][m][g]['yaw_rmse_deg'] for g in groups],width=.24,color=colors[m],label=m)
    axes[1].bar(np.arange(8)+(i-1)*.24,[report['by_route'][r][m]['all']['yaw_rmse_deg'] for r in routes],width=.24,color=colors[m],label=m)
axes[0].set_xticks(range(3),['Near straight <1 deg','Small turn 1-3 deg','Strong turn >=3 deg'])
axes[1].set_xticks(range(8),['lab4','hall4','long7','army5','class5','aspen11','out8','out9'],rotation=30)
for ax in axes: ax.set_ylabel('One-step XYZ z RMSE [deg]'); ax.grid(axis='y',alpha=.25)
axes[0].legend(); fig.tight_layout(); fig.savefig(root/'angle_error_comparison.png',dpi=160)
print(json.dumps(report['pooled'],indent=2))
print('MATCHED_PAIRS',len(rows))
