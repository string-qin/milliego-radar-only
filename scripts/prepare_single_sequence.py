"""Prepare one ColoRadar sequence, radar-only inputs and radar-frame 6DoF targets."""
import json,hashlib,time,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
SEQ="12_21_2020_arpg_lab_run0"
SRC=Path("/storage/data/metaiot_data/public_dataset/coloradar")
OUT=ROOT/"data"/"coloradar_lab0_gap5_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def project(pc):
    xyz=pc[:,:3].astype(float);assert np.isfinite(xyz).all()
    x,y,z=xyz.T;r=np.hypot(x,y);az=np.degrees(np.arctan2(-y,x));el=np.degrees(np.arctan2(z,r))
    ok=(r>0)&(r<30)&(az>=-60)&(az<60)&(el>=-30)&(el<30)
    row=np.clip(np.floor((30-el[ok])/60*64).astype(int),0,63);col=np.floor((az[ok]+60)/120*256).astype(int)
    depth=np.full((64,256),np.inf);np.minimum.at(depth,(row,col),r[ok]);valid=np.isfinite(depth)
    image=np.zeros((64,256),np.uint8);image[valid]=np.floor((30-depth[valid])/30*255).astype(np.uint8)
    return image,int(ok.sum()),int(valid.sum())
def main():
    assert not (OUT/"manifest.json").exists(),"Immutable prepared output already exists"
    OUT.mkdir(parents=True,exist_ok=True)
    seq=SRC/SEQ;t=np.loadtxt(seq/"single_chip/pointclouds/timestamps.txt");gt=np.loadtxt(seq/"groundtruth/timestamps.txt");pose=np.loadtxt(seq/"groundtruth/groundtruth_poses.txt")
    assert len(gt)==len(pose) and np.all(np.diff(t)>0) and np.all(np.diff(gt)>0)
    j=np.clip(np.searchsorted(gt,t),1,len(gt)-1);valid=(t>=gt[0])&(t<=gt[-1])&((gt[j]-gt[j-1])<=.2)
    frame_ids=np.arange(np.flatnonzero(valid)[0],np.flatnonzero(valid)[-1]+1,5);frame_ids=frame_ids[valid[frame_ids]]
    tq=t[frame_ids];Wb=np.tile(np.eye(4),(len(tq),1,1));Wb[:,:3,:3]=Slerp(gt,Rotation.from_quat(pose[:,3:7]))(tq).as_matrix()
    Wb[:,:3,3]=np.stack([np.interp(tq,gt,pose[:,a]) for a in range(3)],1)
    extpath=SRC/"calib/transforms/base_to_single_chip.txt";ext=extpath.read_text().splitlines()
    Bs=np.eye(4);Bs[:3,3]=np.fromstring(ext[0],sep=" ");Bs[:3,:3]=Rotation.from_quat(np.fromstring(ext[1],sep=" ")).as_matrix()
    Ws=Wb@Bs
    n=len(frame_ids);bounds=(int(n*.6),int(n*.8));frame_split=np.where(np.arange(n)<bounds[0],0,np.where(np.arange(n)<bounds[1],1,2))
    pairs=[];deltas=[];labels=[];splits=[]
    for a in range(n-1):
        b=a+1;ia,ib=frame_ids[a],frame_ids[b]
        if ib-ia!=5 or not valid[ia:ib+1].all() or frame_split[a]!=frame_split[b]:continue
        D=np.linalg.inv(Ws[a])@Ws[b]
        pairs.append((a,b));deltas.append(D);labels.append(np.r_[D[:3,3],Rotation.from_matrix(D[:3,:3]).as_euler("xyz",degrees=True)]);splits.append(frame_split[a])
    pairs=np.array(pairs);labels=np.array(labels);splits=np.array(splits);deltas=np.array(deltas)
    # Endpoint disjointness is checked in original radar frame IDs, not pair row numbers.
    sets=[set(frame_ids[np.unique(pairs[splits==s])].tolist()) for s in range(3)]
    assert all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
    recovered=np.tile(np.eye(4),(len(labels),1,1));recovered[:,:3,3]=labels[:,:3];recovered[:,:3,:3]=Rotation.from_euler("xyz",labels[:,3:],degrees=True).as_matrix()
    assert np.max(abs(recovered-deltas))<1e-10
    np.savez_compressed(OUT/"labels.npz",frame_ids=frame_ids,timestamps=tq,world_radar=Ws,pairs=pairs,labels=labels.astype(np.float32),split=splits,deltas=deltas,base_to_radar=Bs)
    def read(i):
        p=seq/("single_chip/pointclouds/data/radar_pointcloud_%d.bin"%i);blob=p.read_bytes();a=np.frombuffer(blob,dtype="<f4")
        assert a.size%5==0;a=a.reshape(-1,5);im,k,occupied=project(a)
        return im,{"frame_id":int(i),"path":str(p),"sha256":hashlib.sha256(blob).hexdigest(),"points":len(a),"in_fov":k,"pixels":occupied}
    images=[];records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for count,(im,rec) in enumerate(pool.map(read,frame_ids)):
            images.append(im);records.append(rec)
            if count%30==0:print("pointclouds",count+1,"/",len(frame_ids),flush=True)
    images=np.stack(images);assert np.all(np.count_nonzero(images,axis=(1,2))>0)
    np.save(OUT/"images.npy",images,allow_pickle=False)
    hist={name:int((splits==s).sum()) for s,name in enumerate(["train","val","test"])}
    config={"sequence":SEQ,"frame_stride":5,"pair_gap_native_frames":5,"nominal_pair_dt_s":.5,"image_hw":[64,256],"azimuth_fov_deg":[-60,60],"elevation_fov_deg":[-30,30],"max_planar_range_m":30,"pixel":"nearest planar range, inverse distance uint8 [0,255]; repeated 3x per frame; first then second frame concatenation","split":"chronological 60/20/20 sampled frames, boundary-crossing pairs excluded; same-sequence validation only","target":"inverse(T_world_radar_a) @ T_world_radar_b; tx ty tz + extrinsic xyz Euler degrees; column-vector transforms","extrinsic_source":str(extpath),"extrinsic_sha256":sha(extpath),"extrinsic_direction_evidence":"ColoRadar_tools-master/python/plot_pointclouds.py builds T_bs and applies T_ws = T_wb @ T_bs","max_gt_bracket_s":.2,"pairs":hist,"frames":len(images),"source_records":records,"label_roundtrip_max_error":float(np.max(abs(recovered-deltas))),"split_frames_disjoint":True,"source_poses_sha256":sha(seq/"groundtruth/groundtruth_poses.txt"),"source_gt_time_sha256":sha(seq/"groundtruth/timestamps.txt"),"source_radar_time_sha256":sha(seq/"single_chip/pointclouds/timestamps.txt"),"images_sha256":sha(OUT/"images.npy"),"labels_sha256":sha(OUT/"labels.npz"),"scope":"Single-sequence reproduction pilot; not original paper dataset or independent-sequence generalization"}
    (OUT/"manifest.json").write_text(json.dumps(config,indent=2));print(json.dumps({"frames":len(images),"pairs":hist,"label_roundtrip_error":config["label_roundtrip_max_error"]},indent=2),flush=True)
if __name__=="__main__":main()

