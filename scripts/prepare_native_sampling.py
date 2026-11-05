"""Restore native 10Hz rather than copying source gap=5 across different sensor frequencies."""
import json,hashlib,concurrent.futures,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scripts.prepare_single_sequence import project
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"data/coloradar_lab0_contractfix_v2";OUT=ROOT/"data/coloradar_lab0_native10_v1"
SRC=Path("/storage/data/metaiot_data/public_dataset/coloradar");SEQ=SRC/"12_21_2020_arpg_lab_run0"
def main():
    OUT.mkdir(exist_ok=False);old=dict(np.load(BASE/"labels.npz"))
    t=np.loadtxt(SEQ/"single_chip/pointclouds/timestamps.txt");gt=np.loadtxt(SEQ/"groundtruth/timestamps.txt");p=np.loadtxt(SEQ/"groundtruth/groundtruth_poses.txt")
    j=np.clip(np.searchsorted(gt,t),1,len(gt)-1);valid=(t>=gt[0])&(t<=gt[-1])&((gt[j]-gt[j-1])<=.2)
    frames=[];splitframes=[];windows=[]
    for s in range(3):
        sample=np.unique(old["pairs"][old["split"]==s]);native=old["frame_ids"][sample];lo,hi=int(native.min()),int(native.max())
        selected=np.arange(lo,hi+1);selected=selected[valid[selected]];frames.extend(selected.tolist());splitframes.extend([s]*len(selected));windows.append([lo,hi])
    frame=np.array(frames);sp=np.array(splitframes);tq=t[frame]
    Wb=np.tile(np.eye(4),(len(frame),1,1));Wb[:,:3,:3]=Slerp(gt,Rotation.from_quat(p[:,3:7]))(tq).as_matrix()
    Wb[:,:3,3]=np.stack([np.interp(tq,gt,p[:,a]) for a in range(3)],1);Ws=Wb@old["base_to_radar"]
    pairs=[];deltas=[];labels=[];split=[]
    for a in range(len(frame)-1):
        b=a+1
        if frame[b]-frame[a]!=1 or sp[a]!=sp[b]:continue
        D=np.linalg.inv(Ws[a])@Ws[b];pairs.append((a,b));deltas.append(D);labels.append(np.r_[D[:3,3],Rotation.from_matrix(D[:3,:3]).as_euler("XYZ",degrees=True)]);split.append(sp[a])
    data={"frame_ids":frame,"timestamps":tq,"world_radar":Ws,"pairs":np.array(pairs),"deltas":np.array(deltas),"labels":np.array(labels,np.float32),"split":np.array(split),"base_to_radar":old["base_to_radar"]}
    np.savez_compressed(OUT/"labels.npz",**data)
    def read(i):
        path=SEQ/("single_chip/pointclouds/data/radar_pointcloud_%d.bin"%i);b=path.read_bytes();pc=np.frombuffer(b,dtype="<f4").reshape(-1,5);im,_,_=project(pc)
        return im,{"frame":int(i),"sha256":hashlib.sha256(b).hexdigest()}
    images=[];records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for n,(im,rec) in enumerate(pool.map(read,frame)):
            images.append(im);records.append(rec)
            if n%100==0:print("READ",n+1,"/",len(frame),flush=True)
    images=np.stack(images);trainframes=sp==0;mean=float((images[trainframes].astype(float)/255).mean())
    normalized=images.astype(np.float32)/255-np.float32(mean)
    np.save(OUT/"images_raw.npy",images,allow_pickle=False);np.save(OUT/"images.npy",normalized,allow_pickle=False)
    # Same physical image and radar pose must match previous trial at every old sampled timestamp.
    lookup={int(x):i for i,x in enumerate(frame)}
    oldimages=np.load(ROOT/"data/coloradar_lab0_gap5_v1/images.npy")
    for i,f in enumerate(old["frame_ids"]):
        if int(f) in lookup:assert np.array_equal(images[lookup[int(f)]],oldimages[i])
    rec={"scope":"Data sampling diagnosis, not exact original projection certification","native_rate_hz":float(1/np.median(np.diff(t))),"previous_effective_hz":2,"sampling_stride":1,"same_physical_split_windows_as_previous_native_frame_ids":windows,"pair_counts":{name:int((data["split"]==s).sum()) for s,name in enumerate(["train","val","test"])},"training_mean":mean,"normalization":"uint8/255 - training_mean","projection":"unchanged direct 64x256 diagnostic projection; normalized; exact original config unresolved","test_evaluation":"Compose 5 adjacent native predictions and evaluate at exactly the previous 47 timestamps, against identical radar GT","source_records":records}
    (OUT/"manifest.json").write_text(json.dumps(rec,indent=2));print(json.dumps({k:v for k,v in rec.items() if k!="source_records"},indent=2))
if __name__=="__main__":main()

