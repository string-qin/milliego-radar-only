"""Evaluation-only attribution; reference substitutions are diagnostics, never model inputs or baseline scores."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path("/home/metaiot_guest/Workspace/qinzx/milliEgo");OUT=ROOT/"runs/expanded_training_v1"
r=json.loads((OUT/"results.json").read_text());d=np.load(ROOT/"data/coloradar_lab0_contractfix_v2/labels.npz");ids=np.flatnonzero(d["split"]==2)
true=d["deltas"][ids]
def integrate(D):
    P=[np.eye(4)]
    for delta in D:P.append(P[-1]@delta)
    return np.stack(P)
reference=integrate(true);source=d["world_radar"];pairs=d["pairs"][ids];frames=np.r_[pairs[0,0],pairs[:,1]];GT=np.linalg.inv(source[frames[0]])@source[frames]
assert abs(reference-GT).max()<1e-9
def ate(D):
    P=integrate(D)
    return float(np.sqrt(np.mean(np.sum((P[:,:3,3]-reference[:,:3,3])**2,axis=1))))
rows=[]
for arm in ["repeated_base","expanded_data"]:
    for seed in r["protocol"]["seeds"]:
        folder=OUT/(arm+"_seed%d"%seed);p=np.load(folder/"full_200_epochs_predictions.npz")["prediction"]
        T=np.tile(np.eye(4),(len(p),1,1));T[:,:3,:3]=Rotation.from_euler("XYZ",p[:,3:],degrees=True).as_matrix();T[:,:3,3]=p[:,:3]
        trans_only=T.copy();trans_only[:,:3,:3]=true[:,:3,:3]
        rot_only=T.copy();rot_only[:,:3,3]=true[:,:3,3]
        residual=T[:,:3,3]-true[:,:3,3]
        terminal_R=Rotation.from_matrix(reference[-1,:3,:3].T@integrate(T)[-1,:3,:3]).magnitude()*180/np.pi
        record={"arm":arm,"seed":seed,"actual_ate_m":ate(T),"diagnostic_perfect_rotation_ate_m":ate(trans_only),"diagnostic_perfect_translation_ate_m":ate(rot_only),
        "mean_local_translation_error_xyz_m":residual.mean(0).tolist(),"predicted_mean_local_translation_xyz_m":p[:,:3].mean(0).tolist(),"true_mean_local_translation_xyz_m":true[:,:3,3].mean(0).tolist(),
        "sum_predicted_step_lengths_m":float(np.linalg.norm(p[:,:3],axis=1).sum()),"sum_true_step_lengths_m":float(np.linalg.norm(true[:,:3,3],axis=1).sum()),"terminal_orientation_error_deg":float(terminal_R)}
        rows.append(record)
summary={}
for arm in ["repeated_base","expanded_data"]:
    rr=[x for x in rows if x["arm"]==arm]
    summary[arm]={k:float(np.mean([x[k] for x in rr])) for k in ["actual_ate_m","diagnostic_perfect_rotation_ate_m","diagnostic_perfect_translation_ate_m","sum_predicted_step_lengths_m","sum_true_step_lengths_m","terminal_orientation_error_deg"]}
    summary[arm]["mean_local_translation_error_xyz_m"]=np.mean([x["mean_local_translation_error_xyz_m"] for x in rr],axis=0).tolist()
report={"scope":"Evaluation-only attribution on previously exposed run0 segment. Perfect-reference substitutions are impossible-oracle diagnostics, NOT deployable algorithm outputs or comparable baseline scores.","summary":summary,"rows":rows}
(OUT/"error_attribution.json").write_text(json.dumps(report,indent=2))
a=r["test_summary"]["repeated_base"]["full_200_epochs"];b=r["test_summary"]["expanded_data"]["full_200_epochs"]
changes={k:(b[k]["mean"]/a[k]["mean"]-1)*100 for k in a}
print(json.dumps({"relative_change_percent":changes,"attribution":summary},indent=2))

