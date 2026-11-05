"""Read-only geometric consistency check. GT is used for diagnostics only, never model inputs."""
import json,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];SRC=Path("/storage/data/metaiot_data/public_dataset/coloradar/12_21_2020_arpg_lab_run0/single_chip/pointclouds/data")
d=np.load(ROOT/"data/coloradar_lab0_contractfix_v2/labels.npz");ids=np.flatnonzero(d["split"]<2)
used=np.unique(d["frame_ids"][d["pairs"][ids]])
def read(i):
    a=np.fromfile(SRC/("radar_pointcloud_%d.bin"%i),dtype="<f4").reshape(-1,5)[:,:3].astype(float)
    valid=np.isfinite(a).all(1)&(np.linalg.norm(a,axis=1)<30)
    return int(i),a[valid]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:cloud=dict(pool.map(read,used))
def distance(a,b):
    if not len(a) or not len(b):return None
    dist=np.r_[cKDTree(a).query(b)[0],cKDTree(b).query(a)[0]]
    return float(np.median(dist))
rows=[]
for j in ids:
    aa,bb=d["frame_ids"][d["pairs"][j]];a,b=cloud[int(aa)],cloud[int(bb)];D=d["deltas"][j];I=np.linalg.inv(D)
    current=distance(a,b);correct=distance(a,b@D[:3,:3].T+D[:3,3]);reverse=distance(a,b@I[:3,:3].T+I[:3,3])
    rows.append({"pair":int(j),"split":"train" if d["split"][j]==0 else "val","uncompensated_m":current,"gt_compensated_m":correct,"inverse_gt_compensated_m":reverse})
out={"scope":"Training+validation only. Sparse radar NN is a noisy diagnostic; it cannot certify exact timing/extrinsics on its own.","pairs":len(rows),"median_nn_uncompensated_m":float(np.median([r["uncompensated_m"] for r in rows])),"median_nn_gt_compensated_m":float(np.median([r["gt_compensated_m"] for r in rows])),"median_nn_inverse_gt_m":float(np.median([r["inverse_gt_compensated_m"] for r in rows])),"fraction_gt_better_than_uncompensated":float(np.mean([r["gt_compensated_m"]<r["uncompensated_m"] for r in rows])),"fraction_gt_better_than_inverse":float(np.mean([r["gt_compensated_m"]<r["inverse_gt_compensated_m"] for r in rows])),"rows":rows}
(ROOT/"reports/geometric_input_consistency.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))

