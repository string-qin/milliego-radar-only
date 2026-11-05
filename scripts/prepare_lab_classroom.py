"""Add lab runs1-4 to training only, preserving run0 evaluation and data representation."""
import json,hashlib,sys,concurrent.futures,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scripts.prepare_single_sequence import project
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"data/coloradar_lab_classroom_v1";SRC=Path("/storage/data/metaiot_data/public_dataset/coloradar");BASE=ROOT/"data/coloradar_lab0_native10_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);base=dict(np.load(BASE/"labels.npz"));records=[]
    for name in ([f"12_21_2020_arpg_lab_run{i}" for i in range(1,5)] + [f"2_23_2021_edgar_classroom_run{i}" for i in (0,1,2,3)]):
        seq=SRC/name;dest=OUT/name
        if (dest/"manifest.json").exists():
            records.append(json.loads((dest/"manifest.json").read_text()));print("REUSE",name,flush=True);continue
        dest.mkdir(exist_ok=True)
        t=np.loadtxt(seq/"single_chip/pointclouds/timestamps.txt");gt=np.loadtxt(seq/"groundtruth/timestamps.txt");p=np.loadtxt(seq/"groundtruth/groundtruth_poses.txt")
        assert np.all(np.diff(t)>0) and np.all(np.diff(gt)>0) and len(gt)==len(p)
        j=np.clip(np.searchsorted(gt,t),1,len(gt)-1);valid=(t>=gt[0])&(t<=gt[-1])&((gt[j]-gt[j-1])<=.2)
        frame=np.flatnonzero(valid);tq=t[frame]
        Wb=np.tile(np.eye(4),(len(frame),1,1));Wb[:,:3,:3]=Slerp(gt,Rotation.from_quat(p[:,3:7]))(tq).as_matrix();Wb[:,:3,3]=np.stack([np.interp(tq,gt,p[:,a]) for a in range(3)],1)
        Ws=Wb@base["base_to_radar"];pairs=np.array([(a,a+1) for a in range(len(frame)-1) if frame[a+1]==frame[a]+1])
        D=np.linalg.inv(Ws[pairs[:,0]])@Ws[pairs[:,1]]
        labels=np.c_[D[:,:3,3],Rotation.from_matrix(D[:,:3,:3]).as_euler("XYZ",degrees=True)].astype(np.float32)
        rr=Rotation.from_euler("XYZ",labels[:,3:].astype(float),degrees=True).as_matrix();assert abs(rr-D[:,:3,:3]).max()<1e-6
        np.savez_compressed(dest/"labels.npz",frame_ids=frame,timestamps=tq,pairs=pairs,labels=labels,deltas=D,world_radar=Ws)
        def read(i):
            path=seq/("single_chip/pointclouds/data/radar_pointcloud_%d.bin"%i);b=path.read_bytes();pc=np.frombuffer(b,dtype="<f4").reshape(-1,5)
            assert np.isfinite(pc).all()
            im,_,_=project(pc)
            return im,{"frame":int(i),"sha256":hashlib.sha256(b).hexdigest()}
        ims=[];sources=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for k,(im,rec) in enumerate(pool.map(read,frame)):
                ims.append(im);sources.append(rec)
                if k%100==0:print("READ",name,k+1,"/",len(frame),flush=True)
        images=np.stack(ims);assert np.all(np.count_nonzero(images,axis=(1,2))>0);np.save(dest/"images_raw.npy",images,allow_pickle=False)
        rec={"sequence":name,"role":"training only","frames":len(images),"pairs":len(pairs),"time_range":[float(tq[0]),float(tq[-1])],"native_hz":float(1/np.median(np.diff(t))),"images_sha256":sha(dest/"images_raw.npy"),"labels_sha256":sha(dest/"labels.npz"),"source_pose_sha256":sha(seq/"groundtruth/groundtruth_poses.txt"),"source_timestamps_sha256":sha(seq/"single_chip/pointclouds/timestamps.txt"),"source_frames":sources}
        (dest/"manifest.json").write_text(json.dumps(rec,indent=2));records.append(rec);print("DONE",name,len(pairs),flush=True)
    # One common normalization scalar for all experimental arms, using training images only.
    bimages=np.load(BASE/"images_raw.npy",mmap_mode="r");train_frames=np.unique(base["pairs"][base["split"]==0])
    total=float(bimages[train_frames].sum(dtype=np.float64));count=int(bimages[train_frames].size)
    for rec in records:
        a=np.load(OUT/rec["sequence"]/"images_raw.npy",mmap_mode="r");total+=float(a.sum(dtype=np.float64));count+=a.size
    mean=json.loads((ROOT/"data/coloradar_lab_expanded_v1/manifest.json").read_text())["common_training_mean"]
    base_roles={s:set((0,int(i)) for i in base["frame_ids"][np.unique(base["pairs"][base["split"]==s])]) for s in range(3)}
    trainset=base_roles[0].copy()
    for k,rec in enumerate(records,1):
        a=np.load(OUT/rec["sequence"]/"labels.npz");trainset.update((k,int(i)) for i in a["frame_ids"])
        assert rec["time_range"][1]<float(base["timestamps"][0]) or rec["time_range"][0]>float(base["timestamps"][-1]),"Recording time overlap with run0"
    assert not trainset&base_roles[1] and not trainset&base_roles[2]
    report={"training_sequences":["run0 fixed original training interval"]+[r["sequence"] for r in records],"added_sequences":len(records),"base_train_pairs":int((base["split"]==0).sum()),"added_train_pairs":sum(r["pairs"] for r in records),"total_train_pairs":int((base["split"]==0).sum())+sum(r["pairs"] for r in records),"base_val_pairs":int((base["split"]==1).sum()),"base_test_pairs":int((base["split"]==2).sum()),"common_training_mean":mean,"normalization":"image/255-common_training_mean; same scalar in expanded and repeated-base controls","normalization_control_note":"Control shares the training-only expanded mean to isolate number/diversity of training examples. It does not see added labels/examples.","split_frame_ids_disjoint":True,"recording_times_disjoint":True,"scope":"Increase training data only; same previously exposed run0 development evaluation, not new unseen-sequence benchmark","projection_status":"unchanged diagnostic projection, exact original config not recovered","sequences":[{k:v for k,v in r.items() if k!="source_frames"} for r in records]}
    (OUT/"manifest.json").write_text(json.dumps(report,indent=2));print("COMPLETE",json.dumps(report),flush=True)
if __name__=="__main__":main()

