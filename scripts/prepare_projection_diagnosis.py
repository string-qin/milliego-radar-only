"""Data-only rendering diagnostics. Fixed FOV/range and fixed model; NOT a recovered original projection config."""
import json,sys,hashlib,concurrent.futures,importlib.util
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/"data/coloradar_lab0_gap5_v1";OUT=ROOT/"data/projection_diagnosis_v1";SRC=Path("/storage/data/metaiot_data/public_dataset/coloradar/12_21_2020_arpg_lab_run0/single_chip/pointclouds/data")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(pc,h,w):
    xyz=pc[:,:3].astype(float);x,y,z=xyz.T;rr=np.hypot(x,y)
    az=np.degrees(np.arctan2(-y,x));el=np.degrees(np.arctan2(z,rr))
    ok=np.isfinite(xyz).all(1)&(rr>0)&(rr<30)&(az>=-60)&(az<60)&(el>=-30)&(el<30)
    row=np.clip(np.floor((30-el[ok])/60*h).astype(int),0,h-1);col=np.floor((az[ok]+60)/120*w).astype(int)
    depth=np.full((h,w),np.inf);np.minimum.at(depth,(row,col),rr[ok]);valid=np.isfinite(depth)
    a=np.zeros((h,w),np.uint8);a[valid]=np.floor((30-depth[valid])/30*255).astype(np.uint8)
    return a
def main():
    OUT.mkdir(exist_ok=False);d=dict(np.load(OLD/"labels.npz"));ids=d["frame_ids"];n=len(ids);cut=int(.6*n)
    # Preserve exact old endpoints/splits. Previous three native frames are the later source tool's mature-buffer rule.
    support={int(i):[int(i)-3,int(i)-2,int(i)-1] for i in ids}
    sets=[]
    for s in range(3):
        sample=np.unique(d["pairs"][d["split"]==s]);sets.append(set(k for i in ids[sample] for k in support[int(i)]))
    assert all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
    needed=sorted(set(ids.tolist())|set(k for values in support.values() for k in values));raw={}
    def read(i):
        p=SRC/("radar_pointcloud_%d.bin"%i);blob=p.read_bytes();a=np.frombuffer(blob,dtype="<f4").reshape(-1,5);assert np.isfinite(a).all()
        return i,a,hashlib.sha256(blob).hexdigest()
    hashes={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,a,h in pool.map(read,needed):raw[i]=a;hashes[str(i)]=h
    images={k:[] for k in ["A_direct_single","B_direct_overlay3","C_coarse_single","D_coarse_overlay3"]}
    for i in ids:
        single=raw[int(i)];stack=np.concatenate([raw[k] for k in support[int(i)]])
        images["A_direct_single"].append(render(single,64,256))
        images["B_direct_overlay3"].append(render(stack,64,256))
        images["C_coarse_single"].append(cv2.resize(render(single,16,64),(256,64),interpolation=cv2.INTER_LINEAR))
        images["D_coarse_overlay3"].append(cv2.resize(render(stack,16,64),(256,64),interpolation=cv2.INTER_LINEAR))
    records={}
    for key,rows in images.items():
        a=np.stack(rows);mean=float((a[:cut].astype(np.float64)/255).mean());normalized=a.astype(np.float32)/255-np.float32(mean)
        np.save(OUT/(key+"_raw.npy"),a,allow_pickle=False);np.save(OUT/(key+".npy"),normalized,allow_pickle=False)
        norm={"training_mean":mean,"training_frames":cut,"only_training_mean":True,"formula":"float32 pixel/255 - float32 training mean"}
        (OUT/(key+"_normalization.json")).write_text(json.dumps(norm,indent=2))
        records[key]={"mean":mean,"raw_nonzero_fraction":float((a!=0).mean()),"normalization":norm,"file":str(OUT/(key+".npy")),"sha256":sha(OUT/(key+".npy"))}
    assert np.array_equal(np.stack(images["A_direct_single"]),np.load(OLD/"images.npy"))
    manifest={"scope":"Test documented missing operations while holding earlier FOV/range fixed; not exact original projection restoration","source":"reference_odombeyondvision/toolkit/radar_preprocess/process_radar_handheld.py","documented_operations":["three adjacent native frames","low-resolution panoramic projection then cv2.resize by4"],"fixed_parameters":{"h_fov":[-60,60],"v_fov":[-30,30],"max_planar_range":30,"collision":"nearest planar distance","raw_grid_coarse":[16,64],"output":[64,256]},"known_unresolved":"Exact original configuration and raster rounding/collision differ; these are declared data-adaptation diagnostics, not certified source-equivalent projection.","support_rule":"overlay mature-buffer source rule [i-3:i], original timestamp i; no GT motion compensation","source_frame_support":support,"support_disjoint_between_splits":True,"original_source_frame_sha256":hashes,"records":records}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(records,indent=2))
if __name__=="__main__":main()

