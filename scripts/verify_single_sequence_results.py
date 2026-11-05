import json
from pathlib import Path
import numpy as np
root=Path("/home/metaiot_guest/Workspace/qinzx/milliEgo");run=root/"runs/pilot_lab0_gap5_v1"
d=np.load(root/"data/coloradar_lab0_gap5_v1/labels.npz");res=json.loads((run/"results.json").read_text())
test=np.load(run/"test_predictions.npz");ind=json.loads((run/"independent_inference.json").read_text())
raw=np.r_[ind["translation_m"],ind["rotation_xyz_degrees"]];error=abs(raw-test["prediction"][0]);assert error.max()<1e-3
maxerr=0
for s in range(3):
    ids=np.flatnonzero(d["split"]==s);last_end=None;P=None;anchor=None
    for i in ids:
        a,b=d["pairs"][i]
        if a!=last_end:P=np.eye(4);anchor=a
        P=P@d["deltas"][i]
        truth=np.linalg.inv(d["world_radar"][anchor])@d["world_radar"][b]
        maxerr=max(maxerr,float(abs(P-truth).max()));last_end=b
assert maxerr<1e-9
pred=np.loadtxt(run/"test_model_segment0.tum");gt=np.loadtxt(run/"test_reference_segment0.tum")
ate=float(np.sqrt(np.mean(np.sum((pred[:,1:4]-gt[:,1:4])**2,axis=1))))
assert abs(ate-res["metrics"]["test"]["model"]["segments"][0]["ate_initial_pose_rmse_m"])<1e-8
out={"raw_pair_inference_vs_cached_prediction_max_abs_error":float(error.max()),"label_integration_vs_radar_gt_max_matrix_error":maxerr,"ate_recomputed_from_tum_m":ate,"test_prediction_mean":test["prediction"].mean(0).tolist(),"test_prediction_std":test["prediction"].std(0).tolist(),"all_checks_passed":True}
(run/"verification.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

