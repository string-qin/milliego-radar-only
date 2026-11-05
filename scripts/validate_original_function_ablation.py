"""Documented preprocessing-operation diagnosis, same algorithm/settings; original projection config still unresolved."""
import json,sys,time,hashlib,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scipy.spatial.transform import Rotation
from milliego.model import RadarOnlyMilliEgo
from scripts.train_single_sequence import aligned_rmse,save_tum
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/original_function_ablation_v1";OLD=ROOT/"data/coloradar_lab0_gap5_v1";NEW=ROOT/"data/coloradar_lab0_contractfix_v2"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lossfn(p,y):return (p[:,:3]-y[:,:3]).square().mean()+.001*(p[:,3:]-y[:,3:]).square().mean()
def metrics(p,d,ids,order):
    T=np.tile(np.eye(4),(len(p),1,1));T[:,:3,3]=p[:,:3];T[:,:3,:3]=Rotation.from_euler(order,p[:,3:],degrees=True).as_matrix()
    true=d["deltas"][ids];error=np.linalg.inv(true)@T
    r=Rotation.from_matrix(error[:,:3,:3]).magnitude()
    rec={"pair_translation_rmse_m":float(np.sqrt(np.mean(np.sum((T[:,:3,3]-true[:,:3,3])**2,1)))),"pair_rotation_rmse_deg":float(np.sqrt(np.mean(np.degrees(r)**2)))}
    pairs=d["pairs"][ids];groups=np.split(np.arange(len(ids)),np.flatnonzero(pairs[1:,0]!=pairs[:-1,1])+1);segs=[];trajs=[]
    for group in groups:
        anchor=pairs[group[0],0];frame=np.r_[anchor,pairs[group,1]]
        G=np.linalg.inv(d["world_radar"][anchor])@d["world_radar"][frame];P=[np.eye(4)]
        for i in group:P.append(P[-1]@T[i])
        P=np.stack(P);ate=np.linalg.norm(P[:,:3,3]-G[:,:3,3],axis=1)
        segs.append({"poses":len(P),"ate_initial_rmse_m":float(np.sqrt(np.mean(ate**2))),"ate_se3_rmse_m":aligned_rmse(P,G),"endpoint_error_m":float(ate[-1]),"path_length_m":float(np.linalg.norm(np.diff(G[:,:3,3],axis=0),axis=1).sum())})
        trajs.append((P,G,d["timestamps"][frame]))
    rec["segments"]=segs
    if len(segs)==1:rec.update({k:segs[0][k] for k in ["ate_initial_rmse_m","ate_se3_rmse_m","endpoint_error_m"]})
    return rec,trajs
def main():
    OUT.mkdir(exist_ok=True);assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    old=dict(np.load(OLD/"labels.npz"));new=dict(np.load(NEW/"labels.npz"))
    for k in ["pairs","split","frame_ids","timestamps","deltas","world_radar"]:assert np.array_equal(old[k],new[k]),k
    ids={name:np.flatnonzero(old["split"]==s) for s,name in enumerate(["train","val","test"])}
    cfg={"seeds":[20261102,20261103,20261104],"epochs":200,"lr":1e-5,"schedule":"constant (published repeated-fit behavior)","batch_size":64,"optimizer":"PyTorch RMSprop alpha=.9 eps=1e-8, no weight decay","loss":"translation MSE m + .001 rotation MSE degrees","gradient_clip":None,"cnn":"official weights frozen","lstm_steps":1,"model_source_sha256":sha(ROOT/"milliego/model.py"),"scope":"Original-function replacement only, same previously exposed segment; fixed non-author-verified FOV/range. No exact original-pipeline claim.","arms":{name:{"normalized":name,"correct_euler":True} for name in ["A_direct_single","B_original_function"]},"checkpoint_selection":"lowest validation loss within each fixed run; no test-driven choice; report all predefined seeds"}
    (OUT/"protocol.json").write_text(json.dumps(cfg,indent=2))
    features={}
    for normalized in cfg["arms"]:
        images=np.load(ROOT/("data/original_function_ablation_v1/"+normalized+".npy"))
        torch.manual_seed(20261102);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt");m.eval();chunks=[]
        with torch.no_grad():
            for batch in np.array_split(old["pairs"],15):
                a=torch.from_numpy(images[batch[:,0]].copy()).cuda().float();b=torch.from_numpy(images[batch[:,1]].copy()).cuda().float()
                x=torch.cat((a[:,None].repeat(1,3,1,1),b[:,None].repeat(1,3,1,1)),1);z=m.cnn(x)
                assert torch.isfinite(z).all();chunks.append(z.detach())
        features[normalized]=torch.cat(chunks);del m
    records=[];pending=[];initial_hash={}
    started=time.monotonic()
    for name,arm in cfg["arms"].items():
        z=features[arm["normalized"]];d=new if arm["correct_euler"] else old;y=torch.from_numpy(d["labels"]).cuda();order="XYZ" if arm["correct_euler"] else "xyz"
        train=torch.as_tensor(ids["train"],device="cuda");val=torch.as_tensor(ids["val"],device="cuda")
        for seed in cfg["seeds"]:
            folder=OUT/(name+"_seed%d"%seed);folder.mkdir(exist_ok=True)
            torch.manual_seed(seed);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt")
            digest=hashlib.sha256()
            for k,v in m.state_dict().items():digest.update(k.encode());digest.update(v.detach().cpu().numpy().tobytes())
            ih=digest.hexdigest()
            if seed in initial_hash:assert initial_hash[seed]==ih,"Initialization differs between arms"
            initial_hash[seed]=ih
            saved=folder/"best.pt"
            if saved.exists():
                checkpoint=torch.load(saved,weights_only=True,map_location="cpu")
                assert checkpoint["arm"]==name and checkpoint["seed"]==seed
                state=checkpoint["model"];selected=checkpoint["selected_epoch"]
                with (folder/"history.csv").open() as f:history=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
                assert len(history)==cfg["epochs"]
                best=history[selected-1]["val_loss"];tl=history[-1]["train_loss"]
                optimizer=None
                print("RESUMED",name,seed,flush=True)
            else:
                optimizer=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=cfg["lr"],alpha=.9,eps=1e-8)
                best=float("inf");history=[];first=None
                for epoch in range(cfg["epochs"]):
                    m.train();shuffled=train[torch.randperm(len(train),device="cuda")]
                    for batch in shuffled.split(cfg["batch_size"]):
                        optimizer.zero_grad(set_to_none=True);p=m.forward_features(z[batch,None])[:,0];L=lossfn(p,y[batch]);assert torch.isfinite(L)
                        L.backward()
                        if epoch in [0,cfg["epochs"]-1]:assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                        optimizer.step()
                    m.eval()
                    with torch.no_grad():
                        pv=m.forward_features(z[val,None])[:,0];pt=m.forward_features(z[train,None])[:,0];vl=float(lossfn(pv,y[val]));tl=float(lossfn(pt,y[train]))
                    history.append({"epoch":epoch+1,"train_loss":tl,"val_loss":vl})
                    if vl<best:
                        best=vl;selected=epoch+1;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            m.load_state_dict(state);m.eval()
            with torch.no_grad():trainp=m.forward_features(z[train,None])[:,0].cpu().numpy();valp=m.forward_features(z[val,None])[:,0].cpu().numpy()
            rec={"arm":name,"seed":seed,"initial_state_sha256":ih,"selected_epoch":selected,"selected_val_loss":best,"train":metrics(trainp,d,ids["train"],order)[0],"val":metrics(valp,d,ids["val"],order)[0],"final_training_loss":tl,"folder":str(folder)}
            # Save predictions on train/val first; defer test evaluation until EVERY arm has finished training.
            np.savez_compressed(folder/"train_val_predictions.npz",train=trainp,val=valp)
            with (folder/"history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
            checkpoint={"model":state,"protocol":cfg,"arm":name,"seed":seed,"selected_epoch":selected,"normalization":json.loads((ROOT/("data/original_function_ablation_v1/"+name+"_normalization.json")).read_text()),"euler_order":order}
            torch.save(checkpoint,folder/"best.pt")
            pending.append((rec,state,d,order,arm["normalized"]))
            print("TRAINED",name,seed,"selected",selected,"val",best,flush=True);del m,optimizer
    checks=[]
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for rec,state,d,order,normalized in pending:
        z=features[normalized];m=RadarOnlyMilliEgo().cuda();m.load_state_dict(state);m.eval()
        test=torch.as_tensor(ids["test"],device="cuda")
        with torch.no_grad():p=m.forward_features(z[test,None])[:,0].cpu().numpy()
        metric,trajs=metrics(p,d,ids["test"],order);rec["test"]=metric
        folder=Path(rec["folder"]);np.savez_compressed(folder/"test_predictions.npz",prediction=p,pair_rows=ids["test"],source_frame_pairs=d["frame_ids"][d["pairs"][ids["test"]]])
        for j,(P,G,t) in enumerate(trajs):
            save_tum(folder/("test_prediction_segment%d.tum"%j),P,t);save_tum(folder/("test_reference_segment%d.tum"%j),G,t)
            diskp=np.loadtxt(folder/("test_prediction_segment%d.tum"%j));diskg=np.loadtxt(folder/("test_reference_segment%d.tum"%j))
            independent=float(np.sqrt(np.mean(np.sum((diskp[:,1:4]-diskg[:,1:4])**2,axis=1))))
            assert abs(independent-metric["segments"][j]["ate_initial_rmse_m"])<1e-8
            if rec["seed"]==cfg["seeds"][0] and j==0:
                ax[0].plot(P[:,0,3],P[:,1,3],label=rec["arm"])
                if rec["arm"]=="A_direct_single":ax[0].plot(G[:,0,3],G[:,1,3],"k--",label="reference")
        if rec["seed"]==cfg["seeds"][0]:
            # Independent raw-input path with training normalization; no GT needed for input.
            pair=d["pairs"][ids["test"][0]];raw=np.load(ROOT/("data/original_function_ablation_v1/"+rec["arm"]+"_raw.npy"))[pair].astype(np.float32)
            if normalized:raw=raw/255-np.float32(json.loads((ROOT/("data/original_function_ablation_v1/"+rec["arm"]+"_normalization.json")).read_text())["training_mean"])
            x=torch.from_numpy(np.concatenate([np.repeat(im[None],3,0) for im in raw],0)[None,None]).cuda()
            with torch.no_grad():rawp=m(x)[0,0].cpu().numpy()
            e=float(abs(rawp-p[0]).max());assert e<1e-3
            loaded=RadarOnlyMilliEgo().cuda();loaded.load_state_dict(torch.load(folder/"best.pt",weights_only=True,map_location="cpu")["model"]);loaded.eval()
            with torch.no_grad():reload=loaded.forward_features(z[test,None])[:,0].cpu().numpy()[0]
            assert np.array_equal(reload,p[0])
            checks.append({"arm":rec["arm"],"raw_input_vs_cache_max_abs_error":e,"reload_equal":True})
            del loaded
        records.append(rec);print("EVALUATED",rec["arm"],rec["seed"],json.dumps(metric),flush=True);del m
    aggregation={}
    names=["pair_translation_rmse_m","pair_rotation_rmse_deg","ate_initial_rmse_m","ate_se3_rmse_m","endpoint_error_m"]
    for arm in cfg["arms"]:
        rr=[r for r in records if r["arm"]==arm]
        aggregation[arm]={key:{"mean":float(np.mean([r["test"][key] for r in rr])),"std":float(np.std([r["test"][key] for r in rr])),"values":[r["test"][key] for r in rr]} for key in names}
    zero=metrics(np.zeros((len(ids["test"]),6)),new,ids["test"],"XYZ")[0]
    trainmean=new["labels"][ids["train"]].mean(0);mean=metrics(np.tile(trainmean,(len(ids["test"]),1)),new,ids["test"],"XYZ")[0]
    keys=["pair_translation_rmse_m","pair_rotation_rmse_deg","ate_initial_rmse_m"]
    for arm in ["A_direct_single","B_original_function"]:
        ax[1].plot(range(len(keys)),[aggregation[arm][k]["mean"] for k in keys],"o-",label=arm)
    ax[1].set_xticks(range(len(keys)),["translation m","rotation deg","ATE m"]);ax[1].set_title("Mean over three fixed seeds; mixed units");ax[1].legend()
    ax[0].set(xlabel="x [m]",ylabel="y [m]",title="Temporal test segment, first fixed seed");ax[0].axis("equal");ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/"comparison.png",dpi=160);plt.close(fig)
    results={"protocol":cfg,"records":records,"aggregate_test_metrics":aggregation,"baselines_corrected_labels":{"zero_motion":zero,"train_mean":mean},"verification":checks,"wall_seconds":time.monotonic()-started,"model_hash_unchanged":sha(ROOT/"milliego/model.py")==cfg["model_source_sha256"],"important_limit":"FOV/range held fixed; raster resolution and documented overlay operations varied. Original exact projection config remains unresolved. Same previously exposed development segment; no unbiased test claim."}
    assert results["model_hash_unchanged"]
    (OUT/"results.json").write_text(json.dumps(results,indent=2));print("COMPLETE",json.dumps({"aggregate":aggregation,"seconds":results["wall_seconds"]}),flush=True)
if __name__=="__main__":main()

