"""Add lab runs1-4 to training only, preserving run0 evaluation and data representation."""
import json,hashlib,sys,concurrent.futures,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scripts.prepare_single_sequence import project
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"data/coloradar_lab_expanded_v1";SRC=Path("/storage/data/metaiot_data/public_dataset/coloradar");BASE=ROOT/"data/coloradar_lab0_native10_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import cv2
    from milliego.preprocess_reference import project_with_original
    old=ROOT/"data/coloradar_lab_expanded_v1"
    new=ROOT/"data/coloradar_source_projection_v1"
    new.mkdir(exist_ok=True)
    manifest=json.loads((old/"manifest.json").read_text())
    sources=[(BASE,new/"base","12_21_2020_arpg_lab_run0")]
    sources += [(old/r["sequence"],new/r["sequence"],r["sequence"]) for r in manifest["sequences"]]
    for source,dest,name in sources:
        dest.mkdir(exist_ok=True)
        if (dest/"images_raw.npy").exists(): continue
        d=np.load(source/"labels.npz")
        frame=d["frame_ids"]
        directory=SRC/name/"single_chip/pointclouds/data"
        needed=sorted({max(0,int(i)-k) for i in frame for k in (1,2,3)})
        def read(i):
            return np.fromfile(directory/f"radar_pointcloud_{i}.bin",dtype="<f4").reshape(-1,5)[:,:3].astype(float)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            points=dict(zip(needed,pool.map(read,needed)))
        images=[]
        for i in frame:
            xyz=np.concatenate([points[max(0,int(i)-k)] for k in (3,2,1)])
            radius=np.hypot(xyz[:,0],xyz[:,1])
            xyz=xyz[np.isfinite(xyz).all(axis=1)&(radius>0)&(radius<30)]
            image=project_with_original(xyz,v_res=float(np.nextafter(60/15,np.inf)),h_res=float(np.nextafter(120/63,np.inf)),v_fov=(-30,30),h_fov=(-60,60),max_v=30,depth=True)
            if image.size==0:image=np.zeros((16,64),np.uint8)
            assert image.shape==(16,64)
            images.append(cv2.resize(image,(256,64),interpolation=cv2.INTER_LINEAR))
        (dest/"labels.npz").symlink_to(source/"labels.npz") if not (dest/"labels.npz").exists() else None
        np.save(dest/"images_raw.npy",np.stack(images))
        print("RENDERED",name,len(images),flush=True)
    # Same centering scalar as prior experiments to isolate rendering.
    (new/"manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
