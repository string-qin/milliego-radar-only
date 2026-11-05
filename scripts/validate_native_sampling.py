"""Sampling-rate diagnosis: fixed model, loss, LR, physical split and common 0.5s evaluation intervals."""
import sys,json,hashlib,time,csv,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scipy.spatial.transform import Rotation
from milliego.model import RadarOnlyMilliEgo
from scripts.validate_projection_diagnosis import metrics,save_tum
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/coloradar_lab0_native10_v1";OUT=ROOT/"runs/native_sampling_diagnosis_v1";OLD=ROOT/"data/coloradar_lab0_contractfix_v2"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loss(p,y):return (p[:,:3]-y[:,:3]).square().mean()+.001*(p[:,3:]-y[:,3:]).square().mean()
def compose_to_common(p,d,old,s):
    lookup={(int(d["frame_ids"][a]),int(d["frame_ids"][b])):j for j,(a,b) in enumerate(d["pairs"])}
    out=[];ids=np.flatnonzero(old["split"]==s)
    for j in ids:
        a,b=old["frame_ids"][old["pairs"][j]];T=np.eye(4)
        for i in range(int(a),int(b)):
            k=lookup[(i,i+1)];delta=np.eye(4);delta[:3,:3]=Rotation.from_euler("XYZ",p[k,3:],degrees=True).as_matrix();delta[:3,3]=p[k,:3];T=T@delta
        out.append(np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_euler("XYZ",degrees=True)])
    return np.array(out),ids
def main():
    OUT.mkdir(exist_ok=False);assert torch.cuda.is_available();torch.set_num_threads(4)
    d=dict(np.load(DATA/"labels.npz"));old=dict(np.load(OLD/"labels.npz"));images=np.load(DATA/"images.npy");pairs=d["pairs"]
    train=torch.as_tensor(np.flatnonzero(d["split"]==0),device="cuda");val=torch.as_tensor(np.flatnonzero(d["split"]==1),device="cuda");y=torch.from_numpy(d["labels"]).cuda()
    batches=math.ceil(len(train)/64);matched_epochs=600//batches
    cfg={"seeds":[20261102,20261103,20261104],"lr":1e-5,"batch_size":64,"epochs":200,"schedule":"constant","no_gradient_clip":True,"cnn_frozen":True,"loss":"MSE trans m + .001 MSE rotation degrees","time_steps":1,"matched_budget_epochs":matched_epochs,"matched_budget_steps":matched_epochs*batches,"full_steps":200*batches,"model_source_sha256":sha(ROOT/"milliego/model.py"),"comparison":"Same 47 test timestamps and 46 old 0.5s intervals; compose native predictions before comparison","scope":"Previously exposed development segment; no untouched-test claim; temporal sampling adaptation; no architecture/hyperparameter search"}
    (OUT/"protocol.json").write_text(json.dumps(cfg,indent=2));print("PROTOCOL",json.dumps(cfg),flush=True)
    torch.manual_seed(20261102);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt");m.eval();chunks=[]
    with torch.no_grad():
        for p in np.array_split(pairs,math.ceil(len(pairs)/16)):
            a=torch.from_numpy(images[p[:,0]].copy()).cuda().float();b=torch.from_numpy(images[p[:,1]].copy()).cuda().float()
            x=torch.cat([a[:,None].repeat(1,3,1,1),b[:,None].repeat(1,3,1,1)],1);chunks.append(m.cnn(x).detach())
    z=torch.cat(chunks);assert torch.isfinite(z).all();del m
    pending=[];start=time.monotonic()
    for seed in cfg["seeds"]:
        torch.manual_seed(seed);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt")
        opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-5,alpha=.9,eps=1e-8);best=float("inf");hist=[];saved={}
        for epoch in range(200):
            m.train();shuffled=train[torch.randperm(len(train),device="cuda")]
            for batch in shuffled.split(64):
                opt.zero_grad(set_to_none=True);pred=m.forward_features(z[batch,None])[:,0];L=loss(pred,y[batch]);assert torch.isfinite(L)
                L.backward()
                if epoch in [0,199]:assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                opt.step()
            m.eval()
            with torch.no_grad():pv=m.forward_features(z[val,None])[:,0];pt=m.forward_features(z[train,None])[:,0];vl=float(loss(pv,y[val]));tl=float(loss(pt,y[train]))
            hist.append({"epoch":epoch+1,"train_loss":tl,"val_loss":vl})
            if vl<best:best=vl;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};bestepoch=epoch+1
            if epoch+1==matched_epochs:saved["matched_updates"]=(state,bestepoch,best)
            if epoch%50==0:print("TRAIN",seed,epoch+1,tl,vl,flush=True)
        saved["same_200_epochs"]=(state,bestepoch,best)
        with (OUT/("history_seed%d.csv"%seed)).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
        for mode,(state,be,bv) in saved.items():
            folder=OUT/(mode+"_seed%d"%seed);folder.mkdir()
            torch.save({"model":state,"config":cfg,"selected_epoch":be,"selected_val_loss":bv,"mode":mode,"seed":seed,"normalization":json.loads((DATA/"manifest.json").read_text())["training_mean"],"euler_order":"XYZ"},folder/"best.pt")
            pending.append((folder,mode,seed,state,be,bv))
        del m,opt
    rows=[]
    # All three fixed-seed models and both fixed update budgets are frozen before evaluating the common test.
    for folder,mode,seed,state,be,bv in pending:
        m=RadarOnlyMilliEgo().cuda();m.load_state_dict(state);m.eval()
        with torch.no_grad():p=torch.cat([m.forward_features(z[batch,None])[:,0].cpu() for batch in torch.arange(len(z),device="cuda").split(64)]).numpy()
        rec={"mode":mode,"seed":seed,"selected_epoch":be,"selected_val_loss":bv}
        for s,name in enumerate(["train","val","test"]):
            prediction,ids=compose_to_common(p,d,old,s);met,trajs=metrics(prediction,old,ids,"XYZ");rec[name]=met
            if name=="test":
                np.savez_compressed(folder/"test_common_predictions.npz",prediction=prediction,pair_rows=ids,native_prediction=p,native_pairs=d["frame_ids"][pairs])
                for j,(P,G,t) in enumerate(trajs):
                    save_tum(folder/("test_prediction_segment%d.tum"%j),P,t);save_tum(folder/("test_reference_segment%d.tum"%j),G,t)
                    diskp=np.loadtxt(folder/("test_prediction_segment%d.tum"%j));diskg=np.loadtxt(folder/("test_reference_segment%d.tum"%j))
                    ate=float(np.sqrt(np.mean(np.sum((diskp[:,1:4]-diskg[:,1:4])**2,1))));assert abs(ate-met["segments"][j]["ate_initial_rmse_m"])<1e-8
        rows.append(rec);print("RESULT",json.dumps(rec),flush=True);del m
    summary={}
    for mode in ["matched_updates","same_200_epochs"]:
        summary[mode]={k:{"mean":float(np.mean([r["test"][k] for r in rows if r["mode"]==mode])),"std":float(np.std([r["test"][k] for r in rows if r["mode"]==mode])),"values":[r["test"][k] for r in rows if r["mode"]==mode]} for k in ["pair_translation_rmse_m","pair_rotation_rmse_deg","ate_initial_rmse_m","ate_se3_rmse_m","endpoint_error_m"]}
    before=json.loads((ROOT/"runs/preprocess_ablation_v1/results.json").read_text())["aggregate_test_metrics"]["D_both_corrected"]
    report={"protocol":cfg,"runs":rows,"test_summary":summary,"reference_2hz_same_200_epochs":before,"model_unchanged":sha(ROOT/"milliego/model.py")==cfg["model_source_sha256"],"seconds":time.monotonic()-start,"input_frames":len(images),"native_pairs":len(pairs)}
    assert report["model_unchanged"]
    (OUT/"results.json").write_text(json.dumps(report,indent=2));print("COMPLETE",json.dumps(summary),flush=True)
if __name__=="__main__":main()

