"""Fix confirmed upstream packing/label contract violations; does NOT certify unknown projection parameters."""
import ast,hashlib,importlib.util,json,math,sys,functools
from pathlib import Path
import numpy as np,h5py
from PIL import Image
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/"data/coloradar_lab0_gap5_v1";OUT=ROOT/"data/coloradar_lab0_contractfix_v2"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(path,name,namespace):
    tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),namespace);return namespace[name]
def main():
    assert not OUT.exists() or not any(OUT.iterdir()),"Immutable corrected output already exists"
    OUT.mkdir(exist_ok=True)
    toolkit=ROOT/"reference_odombeyondvision/toolkit/create_dataset_milliego"
    spec=importlib.util.spec_from_file_location("upstream_euler",ROOT/"upstream/eulerangles.py")
    orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig);orig.reduce=functools.reduce
    namespace={"np":np,"math":math,"SCALER":1.,"RADIUS_2_DEGREE":180/math.pi,"mat2euler":orig.mat2euler}
    source=toolkit/"create_h5_dataset.py";extract(source,"transform44",namespace);official=extract(source,"rotated_to_local",namespace)
    data=dict(np.load(OLD/"labels.npz"));images=np.load(OLD/"images.npy")
    labels=[]
    for a,b in data["pairs"]:
        T=data["world_radar"][[a,b]];q=Rotation.from_matrix(T[:,:3,:3]).as_quat()
        poses=np.c_[data["timestamps"][[a,b]],T[:,:3,3],q]
        labels.append(official(poses)[0])
    labels=np.asarray(labels);assert np.isfinite(labels).all()
    # Independent comparison with original mat2euler, not merely our own encode/decode loop.
    direct=np.array([np.r_[D[:3,3],np.degrees(orig.mat2euler(D[:3,:3]))[::-1]] for D in data["deltas"]])
    official_error=float(abs(direct-labels).max());assert official_error<1e-9
    scipy_label=Rotation.from_matrix(data["deltas"][:,:3,:3]).as_euler("XYZ",degrees=True)
    assert np.max(abs(labels[:,3:]-scipy_label))<1e-9
    reconstructed=np.stack([orig.euler2mat(z=np.radians(y[5]),y=np.radians(y[4]),x=np.radians(y[3])) for y in labels])
    assert np.max(abs(reconstructed-data["deltas"][:,:3,:3]))<1e-10
    n=len(images);cut=int(.6*n);mean=float((images[:cut].astype(np.float64)/255).mean())
    normalized=images.astype(np.float32)/np.float32(255)-np.float32(mean)
    assert abs(float(normalized[:cut].mean()))<1e-7
    # Execute the exact original normalization function, replacing only removed scipy.misc.imread with a grayscale decoder.
    class Misc:
        @staticmethod
        def imread(path):return np.asarray(Image.open(path))
    norm_ns={"np":np,"misc":Misc,"join":lambda *p:str(Path(*p))}
    upstream_norm=extract(source,"load_normalize_1channel_img",norm_ns)
    errors=[]
    for i in [0,cut-1,cut,n-1]:
        name="normalization_reference_%d.png"%i;Image.fromarray(images[i]).save(OUT/name)
        ref=upstream_norm("0,255",repr(mean),str(OUT),name)
        err=float(abs(ref[0,:,:,0]-normalized[i]).max());assert err==0;errors.append(err)
    np.save(OUT/"images_raw_projection.npy",images,allow_pickle=False)
    np.save(OUT/"images_normalized.npy",normalized,allow_pickle=False)
    data["labels"]=labels.astype(np.float32);np.savez_compressed(OUT/"labels.npz",**data)
    mean_json={"normalization":"image.astype(float32)/255 - training_mean","training_mean":mean,"mean_source":"first 60 percent of sampled frames, training only","training_frame_count":cut,"test_or_val_used_for_mean":False}
    (OUT/"normalization.json").write_text(json.dumps(mean_json,indent=2))
    records=[]
    for s,name in enumerate(["train","val","test"]):
        ids=np.flatnonzero(data["split"]==s);pairs=data["pairs"][ids]
        breaks=np.flatnonzero(pairs[1:,0]!=pairs[:-1,1])+1
        for j,indices in enumerate(np.split(ids,breaks)):
            frame=np.r_[data["pairs"][indices[0],0],data["pairs"][indices,1]]
            path=OUT/("%s_segment%d.h5"%(name,j))
            with h5py.File(path,"w") as f:
                f.create_dataset("mmwave_middle_data",data=normalized[frame,None,:,:,None],compression="gzip")
                f.create_dataset("label_data",data=labels[indices][None].astype(np.float32))
                f.create_dataset("timestamp",data=data["timestamps"][frame][None])
                f.create_dataset("source_radar_frame_id",data=data["frame_ids"][frame])
                f.attrs["baseline_ready"]=False;f.attrs["projection_status"]="v1 custom projection retained for isolating confirmed fixes; not certified"
            with h5py.File(path) as f:
                assert f["mmwave_middle_data"].shape==(len(frame),1,64,256,1)
                assert f["label_data"].shape==(1,len(frame)-1,6)
            records.append({"path":str(path),"split":name,"frames":len(frame),"pairs":len(indices)})
    report={"status":"Two confirmed data-contract violations corrected; projection configuration still unresolved; NOT formal baseline-ready",
      "normalization":mean_json,"official_label_max_abs_error":official_error,
      "normalization_exact_source_function_errors":errors,
      "old_vs_official_rotation_label_max_abs_difference_deg":float(abs(labels[:,3:]-np.load(OLD/"labels.npz")["labels"][:,3:]).max()),
      "normalized_min_max":[float(normalized.min()),float(normalized.max())],
      "rotation_convention":"official mat2euler returns z,y,x; labels store x,y,z degrees; equivalent to scipy intrinsic XYZ, NOT lowercase extrinsic xyz",
      "source_packager":str(source),"source_packager_sha256":sha(source),"source_upstream_euler_sha256":sha(ROOT/"upstream/eulerangles.py"),
      "reference_repository":"mfkiwl/OdomBeyondVision preserved copy, not a verified byte-identical snapshot of the removed MAPS-Lab repository",
      "reference_commit":"d30b06427457bf65a06c9c52245874d736b585fa",
      "unchanged":["milliego/model.py","weights","upstream","v1 input files and labels","previous training/test results"],
      "not_applied":["no learning-rate tuning","no rotation-output multiplier","no CNN unfreezing","no architecture modifications","no new training or test selection"],
      "pending":["recover exact h_fov/v_fov/h_res/v_res/max_v/mmwave_dist_thre from original config or processed sample",
      "reconcile planar distance in original repository projection with 3D distance in later toolkit",
      "reproduce adjacent-frame overlay and OpenCV linear resize with certified parameters; old direct 64x256 projection remains exploratory",
      "verify original-vs-PyTorch normalization+label and full projection contract before official baseline run"],
      "hdf5":records}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()

