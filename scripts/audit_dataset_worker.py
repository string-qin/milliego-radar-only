"""Read-only metadata and sampled-payload audit; writes only inside milliEgo/reports."""
import json,os,re,time,traceback
from pathlib import Path
from decimal import Decimal
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
SRC=Path("/storage/data/metaiot_data/public_dataset")
REPORT=ROOT/"reports"
out={"scope":"All-sequence metadata; sampled payloads, not all-frame integrity or calibration certification.","datasets":{},"errors":{}}
def save():
    (REPORT/"dataset_audit.json").write_text(json.dumps(out,indent=2,allow_nan=False))
def run(name,fn):
    print("START",name,flush=True)
    try:out["datasets"][name]=fn()
    except Exception:out["errors"][name]=traceback.format_exc()
    save(); print("DONE",name,flush=True)
def stats(a):
    a=np.asarray(a)
    return {"shape":list(a.shape),"dtype":str(a.dtype),"finite":bool(np.isfinite(a).all()),"min":float(np.nanmin(a)) if a.size else None,"max":float(np.nanmax(a)) if a.size else None}
def ts_stats(t):
    d=np.diff(t)
    return {"n":len(t),"first":float(t[0]),"last":float(t[-1]),"nonpositive_dt":int((d<=0).sum()),"dt_min_median_max":np.quantile(d,[0,.5,1]).tolist() if len(d) else []}
def nums(p):return np.loadtxt(p,ndmin=1)
def idx(p):return int(re.findall(r"\d+",Path(p).stem)[-1])
def pano(points):
    xyz=np.asarray(points)[:,:3].astype(float);finite=np.isfinite(xyz).all(axis=1);xyz=xyz[finite]
    x,y,z=xyz.T;r=np.hypot(x,y)
    az=np.degrees(np.arctan2(-y,x));el=np.degrees(np.arctan2(z,r))
    valid=(r>0)&(r<30)&(az>=-60)&(az<60)&(el>=-30)&(el<30)
    row=np.floor((30-el[valid])/60*64).astype(int);col=np.floor((az[valid]+60)/120*256).astype(int)
    row=np.clip(row,0,63);col=np.clip(col,0,255)
    depth=np.full((64,256),np.inf,np.float32);np.minimum.at(depth,(row,col),r[valid])
    image=np.zeros_like(depth);ok=np.isfinite(depth);image[ok]=(30-depth[ok])/30
    return image,{"source_points":len(points),"finite_points":len(xyz),"points_in_candidate_fov":int(valid.sum()),"occupied_pixels":int(ok.sum())}
def coloradar():
    base=SRC/"coloradar";seqs=[base/os.environ["AUDIT_SEQUENCE"]] if os.environ.get("AUDIT_SEQUENCE") else sorted(p for p in base.iterdir() if (p/"groundtruth/timestamps.txt").is_file())
    rows=[];samples=[]
    for seq in seqs:
        t=nums(seq/"single_chip/pointclouds/timestamps.txt");g=nums(seq/"groundtruth/timestamps.txt")
        pose=nums(seq/"groundtruth/groundtruth_poses.txt")
        files=sorted((seq/"single_chip/pointclouds/data").glob("*.bin"),key=idx);ids=np.array([idx(p) for p in files])
        j=np.searchsorted(g,t);j=np.clip(j,1,len(g)-1)
        valid=(t>=g[0])&(t<=g[-1])&((g[j]-g[j-1])<=.2)
        fset=set(ids.tolist());present=np.array([i in fset for i in range(len(t))])
        valid &= present
        pairs=valid[:-1]&valid[1:]&(np.diff(t)>0)&(np.diff(t)<=.2)
        file=files[len(files)//2]
        pc=np.empty((0,5),dtype=np.float32);ps={"not_sampled":"metadata-only sequence"}
        if os.environ.get("AUDIT_PAYLOAD")=="1":
            pc=np.fromfile(file,dtype="<f4").reshape(-1,5);im,ps=pano(pc)
        rows.append({"sequence":seq.name,"radar":ts_stats(t),"gt":ts_stats(g),"pose_rows":len(pose),
          "pose_finite":bool(np.isfinite(pose).all()),"quaternion_norm_max_error":float(abs(np.linalg.norm(pose[:,3:7],axis=1)-1).max()),
          "radar_files":len(files),"missing_indices":int((~present).sum()),"files_outside_timestamp_index":int(((ids<0)|(ids>=len(t))).sum()),
          "candidate_frames_bracketed_max_gt_gap_0_2s":int(valid.sum()),"candidate_adjacent_pairs_max_dt_0_2s":int(pairs.sum()),
          "sample_file":str(file),"sample":stats(pc),"projection":ps})
        if seq.name=="12_21_2020_arpg_lab_run0":
            chosen=np.flatnonzero(pairs);k=int(chosen[len(chosen)//2]);byid={idx(p):p for p in files}
            for i in range(k,k+4):
                a=np.fromfile(byid[i],dtype="<f4").reshape(-1,5);image,_=pano(a);samples.append(image)
            np.savez_compressed(REPORT/"coloradar_diagnostic_inputs.npz",images=np.stack(samples),timestamps=t[k:k+4])
            # Interpolated BASE-frame labels are for pipeline diagnostics only; radar extrinsic is NOT silently assumed.
            tq=t[k:k+4];R=Slerp(g,Rotation.from_quat(pose[:,3:7]))(tq).as_matrix()
            p=np.stack([np.interp(tq,g,pose[:,a]) for a in range(3)],axis=1)
            delta_t=np.einsum("nij,nj->ni",R[:-1].transpose(0,2,1),p[1:]-p[:-1])
            delta_R=R[:-1].transpose(0,2,1)@R[1:]
            labels=np.c_[delta_t,Rotation.from_matrix(delta_R).as_euler("xyz")]
            np.save(REPORT/"coloradar_base_frame_diagnostic_labels.npy",labels,allow_pickle=False)
    cal={p.name:p.read_text() for p in (base/"calib/transforms").glob("*.txt")}
    return {"sequence_count":len(rows),"radar_timestamp_total":sum(r["radar"]["n"] for r in rows),
      "candidate_frames_total":sum(r["candidate_frames_bracketed_max_gt_gap_0_2s"] for r in rows),
      "candidate_pairs_total":sum(r["candidate_adjacent_pairs_max_dt_0_2s"] for r in rows),
      "sampled_payloads":sum(int(r["sample"]["shape"][0]>0) for r in rows),"transforms_raw":cal,"sequences":rows,
      "projection_contract":"Diagnostic only: 64x256, azimuth atan2(-y,x) [-60,60), elevation [-30,30), planar distance<30m, nearest collision, values [0,1]. Not a claimed original-data preprocessing match.",
      "label_status":"BASE-frame interpolation diagnostically constructed; official base/sensor extrinsic direction must be verified before radar-frame supervision."}
def kradar():
    base=SRC/"KRadar";seqs=[base/os.environ["AUDIT_SEQUENCE"]] if os.environ.get("AUDIT_SEQUENCE") else sorted((p for p in base.iterdir() if p.name.isdigit()),key=lambda p:int(p.name));rows=[]
    for seq in seqs:
        labels=sorted((seq/"info_label").glob("*.txt"));gtpath=base/("resources/odometry/gt/gt_%02d.txt"%int(seq.name))
        g=np.loadtxt(gtpath).reshape(-1,3,4);R=g[:,:,:3]
        sample=[]
        for f in [labels[0],labels[len(labels)//2],labels[-1]]:
            header=f.open().readline().strip();m=re.search(r"=([\d_]+),\s*timestamp=([\d.]+)",header);ids=list(map(int,m[1].split("_")))
            sample.append({"file":f.name,"radar_index":ids[0],"lidar_index":ids[1],"timestamp":m[2],
              "radar_exists":(seq/("radar_tesseract/tesseract_%05d.mat"%ids[0])).is_file()})
        rows.append({"sequence":seq.name,"label_count":len(labels),"gt_count":len(g),"counts_match":len(labels)==len(g),"gt_finite":bool(np.isfinite(g).all()),
          "rotation_orthogonality_max_error":float(abs(R.transpose(0,2,1)@R-np.eye(3)).max()),"rotation_det_max_error":float(abs(np.linalg.det(R)-1).max()),
          "sampled_mapping":sample,"radar_tensor_count":len(list((seq/"radar_tesseract").glob("*.mat")))})
    if os.environ.get("AUDIT_PAYLOAD")!="1":return {"sequences":rows,"sequence_count":len(rows)}
    axes=loadmat(base/"resources/info_arr.mat")
    axis={k:stats(v) for k,v in axes.items() if not k.startswith("__")}
    rawpath=base/"1/radar_tesseract/tesseract_00033.mat";start=time.monotonic()
    raw=loadmat(rawpath,variable_names=["arrDREA"])["arrDREA"]
    rawstats=stats(raw);power=raw.mean(axis=0);del raw
    # Diagnostic point proposal, NOT final calibrated CFAR or sensor coordinates.
    from scipy.ndimage import maximum_filter
    peaks=(power==maximum_filter(power,size=3))&(power>np.quantile(power,.995))
    ijk=np.argwhere(peaks);strength=power[peaks];order=np.argsort(strength)[-4096:];ijk=ijk[order]
    rr=axes["arrRange"].ravel()[ijk[:,0]];el=np.radians(axes["arrElevation"].ravel()[ijk[:,1]]);az=np.radians(axes["arrAzimuth"].ravel()[ijk[:,2]])
    xyz=np.c_[rr*np.cos(el)*np.cos(az),-rr*np.cos(el)*np.sin(az),rr*np.sin(el)]
    image,ps=pano(xyz);np.savez_compressed(REPORT/"kradar_diagnostic_projection.npz",image=image,xyz_candidate=xyz)
    return {"sequences":rows,"sequence_count":len(rows),"label_total":sum(r["label_count"] for r in rows),"axes":axis,
      "raw_sample":{"path":str(rawpath),"array":rawstats,"read_reduce_seconds":time.monotonic()-start,"candidate_points":len(xyz),"projection":ps},
      "remaining":"Only 3 label headers per sequence checked. Native radar acquisition timestamps and full radar-to-reference extrinsic not certified. Peak extraction and elevation sign are diagnostic candidates, not a validated final point cloud.",
      "reference":"Official LeGO-LOAM odometry, not independent perfect ground truth."}
def read_messages(path,limit=None):
    with Path(path).open() as f:
        for line in f:
            s=line.strip().rstrip(",")
            if not s.startswith('{"header"') and not s.startswith('{"_timestamp_ns"'):continue
            try:yield json.loads(s)
            except json.JSONDecodeError:continue
def radelft():
    rows=[]
    for scene in ([SRC/"RaDelft"/os.environ["AUDIT_SEQUENCE"]] if os.environ.get("AUDIT_SEQUENCE") else sorted((SRC/"RaDelft").glob("Scene*"))):
        ros=scene/"rosDS";files=sorted((ros/"radar_ososos").glob("*.npy"))
        tns=np.array([int(Decimal(p.stem)*Decimal(10**9)) for p in files],dtype=np.int64);t=tns/1e9
        messages=list(read_messages(ros/"gps_odom.topic.json"))
        gt=np.array([m["_timestamp_ns"] for m in messages],dtype=np.int64)
        pos=np.array([[m["pose"]["pose"]["position"][a] for a in ("x","y","z")] for m in messages])
        quat=np.array([[m["pose"]["pose"]["orientation"][a] for a in ("x","y","z","w")] for m in messages])
        j=np.clip(np.searchsorted(gt,tns),1,len(gt)-1)
        valid=(tns>=gt[0])&(tns<=gt[-1])&((gt[j]-gt[j-1])<=100000000)
        sm=[]
        for f in [files[0],files[len(files)//2],files[-1]]:
            a=np.load(f,allow_pickle=False);item={"file":f.name,"shape":list(a.shape),"dtype":str(a.dtype),"fields":a.dtype.names}
            if a.dtype.names:
                xyz=np.stack([a[x].ravel() for x in ("x","y","z")],axis=-1)
            else:xyz=a.reshape(-1,a.shape[-1])[:,:3]
            im,ps=pano(xyz);item.update({"xyz":stats(xyz),"projection":ps});sm.append(item)
            if scene.name=="Scene1" and f==files[len(files)//2]:np.savez_compressed(REPORT/"radelft_diagnostic_projection.npz",image=im)
        tf_seen={};n=0
        for m in read_messages(ros/"tf.topic.json"):
            n+=1
            for tr in m.get("transforms",[]):
                a=tr.get("tform",[])
                if a:tf_seen[tr.get("frame_id","unknown")]=a
            if n>=2000:break
        mat=loadmat(scene/"RadarCubes/timestamps.mat")
        matstats={k:stats(v) for k,v in mat.items() if not k.startswith("__") and np.issubdtype(v.dtype,np.number)}
        radar_topic_present=(ros/"radar.topic.json").is_file()
        first=next(read_messages(ros/"radar.topic.json")) if radar_topic_present else {"_timestamp_ns":None}
        rows.append({"scene":scene.name,"radar":ts_stats(t),"odom":ts_stats(gt/1e9),"radar_npy_count":len(files),"odom_count":len(gt),
          "candidate_time_covered_frames_max_gt_gap_0_1s":int(valid.sum()),"odom_position_min":pos.min(0).tolist(),"odom_position_max":pos.max(0).tolist(),
          "odom_quaternion_min":quat.min(0).tolist(),"odom_quaternion_max":quat.max(0).tolist(),"odom_finite":bool(np.isfinite(pos).all() and np.isfinite(quat).all()),
          "cube_time_arrays":matstats,"radar_topic_first_timestamp_ns":first["_timestamp_ns"],"radar_topic_present":radar_topic_present,"npy_first_minus_radar_topic_first_s":float((tns[0]-first["_timestamp_ns"])/1e9) if radar_topic_present else None,
          "tf_messages_sampled":n,"tf_nonempty_transforms_sampled":tf_seen,"samples":sm})
    return {"scenes":rows,"scene_count":len(rows),"radar_frames_total":sum(r["radar_npy_count"] for r in rows),
      "candidate_time_covered_total":sum(r["candidate_time_covered_frames_max_gt_gap_0_1s"] for r in rows),
      "remaining":"Time overlap is NOT synchronization certification; exported NPY and original radar JSON clocks differ. Full radar-to-an_device transform is unverified. z/roll/pitch must be inspected before claiming 6DoF supervision. radar_gt NPY are grids, not odometry."}
if __name__=="__main__":
    fn={"ColoRadar":coloradar,"K-Radar":kradar,"RaDelft":radelft}[os.environ["AUDIT_KIND"]]
    result=fn()
    Path(os.environ["AUDIT_OUTPUT"]).write_text(json.dumps(result,indent=2,allow_nan=False))

