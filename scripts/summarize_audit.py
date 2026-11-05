import json,collections
from pathlib import Path
root=Path("/home/metaiot_guest/Workspace/qinzx/milliEgo")
index=json.loads((root/"reports/dataset_audit_index.json").read_text())
out={"completed":index["completed"],"total":index["attempted_total"],"status_counts":dict(collections.Counter(r["dataset"]+":"+r["status"] for r in index["records"]))}
col=[];kr=[];ra=[]
for r in index["records"]:
    if r["status"]!="passed":continue
    d=json.loads(Path(r["result"]).read_text())
    if r["dataset"]=="ColoRadar":col+=d["sequences"]
    elif r["dataset"]=="K-Radar":kr+=d["sequences"]
    else:ra+=d["scenes"]
out["coloradar"]={"sequences":len(col),"radar_frames":sum(x["radar"]["n"] for x in col),"candidate_frames":sum(x["candidate_frames_bracketed_max_gt_gap_0_2s"] for x in col),"candidate_pairs":sum(x["candidate_adjacent_pairs_max_dt_0_2s"] for x in col),"missing_indices":sum(x["missing_indices"] for x in col),"nonpositive_radar_dt":sum(x["radar"]["nonpositive_dt"] for x in col),"nonpositive_gt_dt":sum(x["gt"]["nonpositive_dt"] for x in col),"max_q_norm_error":max(x["quaternion_norm_max_error"] for x in col),"payload_samples":[{"seq":x["sequence"],"shape":x["sample"]["shape"],"projection":x["projection"]} for x in col if x["sample"]["shape"][0]>0]}
out["kradar"]={"sequences":len(kr),"labels":sum(x["label_count"] for x in kr),"gt":sum(x["gt_count"] for x in kr),"radar_tensors":sum(x["radar_tensor_count"] for x in kr),"all_counts_match":all(x["counts_match"] for x in kr),"all_sampled_radar_exists":all(t["radar_exists"] for x in kr for t in x["sampled_mapping"]),"max_rotation_error":max(x["rotation_orthogonality_max_error"] for x in kr)}
out["radelft"]={"scenes":len(ra),"radar_frames":sum(x["radar_npy_count"] for x in ra),"candidate_frames":sum(x["candidate_time_covered_frames_max_gt_gap_0_1s"] for x in ra),"details":[{"scene":x["scene"],"frames":x["radar_npy_count"],"covered":x["candidate_time_covered_frames_max_gt_gap_0_1s"],"position_z_range":[x["odom_position_min"][2],x["odom_position_max"][2]],"quaternion_xy_minmax":[x["odom_quaternion_min"][:2],x["odom_quaternion_max"][:2]],"tf_nonempty_keys":list(x["tf_nonempty_transforms_sampled"]),"radar_topic_present":x.get("radar_topic_present",True),"clock_diff":x["npy_first_minus_radar_topic_first_s"],"samples":[{"shape":s["shape"],"dtype":s["dtype"],"projection":s["projection"]} for s in x["samples"]]} for x in ra]}
(root/"reports/dataset_summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

