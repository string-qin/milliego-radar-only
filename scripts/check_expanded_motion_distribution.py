import json
from pathlib import Path
import numpy as np
root=Path("/home/metaiot_guest/Workspace/qinzx/milliEgo");data=root/"data/coloradar_lab_expanded_v1";base=root/"data/coloradar_lab0_native10_v1";out=root/"runs/expanded_training_v1"
d=np.load(base/"labels.npz");manifest=json.loads((data/"manifest.json").read_text());rows=[];alllabels=[];allspeeds=[]
def summarize(name,a,ids):
    pair=a["pairs"][ids];dt=a["timestamps"][pair[:,1]]-a["timestamps"][pair[:,0]]
    y=a["labels"][ids];speed=np.linalg.norm(y[:,:3],axis=1)/dt
    record={"name":name,"pairs":len(y),"mean_local_translation_xyz_m":y[:,:3].mean(0).tolist(),"speed_mean_mps":float(speed.mean()),"speed_p10_p50_p90_mps":np.quantile(speed,[.1,.5,.9]).tolist(),"fraction_speed_under_0_1mps":float((speed<.1).mean())}
    return record,y,speed
rec,y,s=summarize("base_train",d,np.flatnonzero(d["split"]==0));rows.append(rec);alllabels.append(y);allspeeds.append(s)
for record in manifest["sequences"]:
    a=np.load(data/record["sequence"]/"labels.npz");rec,y,s=summarize(record["sequence"],a,np.arange(len(a["labels"])));rows.append(rec);alllabels.append(y);allspeeds.append(s)
for split,name in [(1,"validation"),(2,"evaluation")]:
    rec,_,_=summarize(name,d,np.flatnonzero(d["split"]==split));rows.append(rec)
y=np.concatenate(alllabels);s=np.concatenate(allspeeds);summary={"expanded_training_mean_local_translation_xyz_m":y[:,:3].mean(0).tolist(),"expanded_training_speed_mean_mps":float(s.mean()),"expanded_training_fraction_speed_under_0_1mps":float((s<.1).mean()),"rows":rows,"scope":"Motion-distribution diagnostic only; no reweighting, data filtering, model changes, or new test-driven model selection."}
(out/"motion_distribution.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

