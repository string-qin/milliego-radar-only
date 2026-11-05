"""Single-sequence pilot: fixed pretrained radar CNN, original-size recurrent regressor, radar-only labels."""
import argparse,copy,csv,json,hashlib,math,sys,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scipy.spatial.transform import Rotation
from milliego.model import RadarOnlyMilliEgo,KerasLSTM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/coloradar_lab0_gap5_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lossfn(pred,target):return (pred[...,:3]-target[...,:3]).square().mean()+.001*(pred[...,3:]-target[...,3:]).square().mean()
def invariant_checks():
    cell=KerasLSTM(3,4).double().eval()
    rng=np.random.default_rng(4);x=rng.normal(size=(2,3,3));h=np.zeros((2,4));c=h.copy();out=[]
    w=cell.kernel.detach().numpy();u=cell.recurrent.detach().numpy();b=cell.bias.detach().numpy()
    for t in range(3):
        z=x[:,t]@w.T+h@u.T+b;i,f,g,o=np.split(z,4,-1)
        i=np.clip(.2*i+.5,0,1);f=np.clip(.2*f+.5,0,1);o=np.clip(.2*o+.5,0,1)
        c=f*c+i*np.tanh(g);h=o*np.tanh(c);out.append(h)
    ref=np.stack(out,1);actual=cell(torch.from_numpy(x)).detach().numpy();error=float(abs(ref-actual).max());assert error<1e-12
    # Lever-arm regression: a pure base rotation must move a noncentral radar origin.
    A=np.eye(4);A[:3,:3]=Rotation.from_euler("z",90,degrees=True).as_matrix();B=np.eye(4);B[:3,3]=[1,0,0]
    delta=np.linalg.inv(B)@A@B;assert np.allclose(delta[:3,3],[-1,1,0])
    return {"keras_lstm_numpy_max_abs_error":error,"lever_arm_pure_rotation_check":True}
def cache_features(model,d,images):
    cache=DATA/"frozen_cnn_features.pt";provenance={"model_source_sha256":sha(ROOT/"milliego/model.py"),"cnn_sha256":sha(ROOT/"weights/cnn_torch.pt"),"images_sha256":sha(DATA/"images.npy"),"labels_sha256":sha(DATA/"labels.npz"),"input_scale":"raw uint8 cast to float32 [0,255]"}
    if cache.exists():
        payload=torch.load(cache,weights_only=True,map_location="cpu");assert payload["provenance"]==provenance
        return payload["features"]
    chunks=[];pairs=d["pairs"]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(pairs),16):
            p=pairs[start:start+16];a=torch.from_numpy(images[p[:,0]].copy()).cuda().float();b=torch.from_numpy(images[p[:,1]].copy()).cuda().float()
            x=torch.cat((a[:,None].repeat(1,3,1,1),b[:,None].repeat(1,3,1,1)),1)
            z=model.cnn(x);assert z.shape[1:]==(1024,1,4) and torch.isfinite(z).all();chunks.append(z.cpu())
    features=torch.cat(chunks);torch.save({"features":features,"provenance":provenance},cache);return features
def predict(model,z,indices):
    model.eval()
    with torch.no_grad():return torch.cat([model.forward_features(z[ids,None])[:,0].cpu() for ids in torch.as_tensor(indices,device="cuda").split(64)]).numpy()
def fit_overfit(z,y,indices,steps,output):
    torch.manual_seed(123);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt");m.eval()
    opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-3,alpha=.9,eps=1e-8)
    small=np.array(indices)[np.linspace(0,len(indices)-1,min(16,len(indices)),dtype=int)];ids=torch.as_tensor(small,device="cuda")
    target=y[ids]
    with torch.no_grad():initial=float(lossfn(m.forward_features(z[ids,None])[:,0],target))
    hist=[];best=float("inf")
    for step in range(steps):
        opt.zero_grad(set_to_none=True);pred=m.forward_features(z[ids,None])[:,0];loss=lossfn(pred,target)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
        if step%25==0 or step==steps-1:
            with torch.no_grad():value=float(lossfn(m.forward_features(z[ids,None])[:,0],target))
            hist.append({"step":step+1,"loss":value});best=min(best,value)
    rec={"pairs":small.tolist(),"steps":steps,"dropout_disabled_for_diagnostic":True,"initial_loss":initial,"best_loss":best,"fraction_reduction":1-best/initial,"passed_90_percent_reduction":best<initial*.1,"learning_rate":1e-3,"scope":"Training-set sanity test only, excluded from model selection; discarded diagnostic model"}
    (output/"overfit.json").write_text(json.dumps(rec,indent=2));(output/"overfit_history.json").write_text(json.dumps(hist,indent=2));print("OVERFIT",json.dumps(rec),flush=True)
    del m,opt;torch.cuda.empty_cache();return rec
def fit(z,y,split,lr,epochs,output):
    torch.manual_seed(20261102);model=RadarOnlyMilliEgo().cuda();model.load_cnn(ROOT/"weights/cnn_torch.pt")
    opt=torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad],lr=lr,alpha=.9,eps=1e-8)
    train=torch.as_tensor(np.flatnonzero(split==0),device="cuda");val=torch.as_tensor(np.flatnonzero(split==1),device="cuda")
    best=float("inf");beststate=None;history=[];started=time.monotonic()
    with torch.no_grad():
        model.eval();initial_val=float(lossfn(model.forward_features(z[val,None])[:,0],y[val]))
    for epoch in range(epochs):
        model.train();rate=lr*.75**((epoch+1)//25)
        for group in opt.param_groups:group["lr"]=rate
        shuffled=train[torch.randperm(len(train),device="cuda")];total=0
        for ids in shuffled.split(32):
            opt.zero_grad(set_to_none=True);pred=model.forward_features(z[ids,None])[:,0];loss=lossfn(pred,y[ids])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            vl=float(lossfn(model.forward_features(z[val,None])[:,0],y[val]))
            tl=float(lossfn(model.forward_features(z[train,None])[:,0],y[train]))
        history.append({"epoch":epoch+1,"lr":rate,"train_dropout_loss":total/len(train),"train_eval_loss":tl,"val_loss":vl})
        if vl<best:
            best=vl;bestepoch=epoch+1;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%25==0 or epoch==epochs-1:print("TRAIN",lr,epoch+1,"train",round(tl,7),"val",round(vl,7),flush=True)
    path=output/("lr_%g"%lr);path.mkdir()
    with (path/"history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    config={"learning_rate":lr,"epochs":epochs,"selected_epoch":bestepoch,"best_val_loss":best,"initial_val_loss":initial_val,"cnn_frozen":True,"optimizer":"RMSprop alpha=.9 eps=1e-8","loss":"MSE_translation_m + .001*MSE_rotation_degrees","schedule":"lr*.75**floor((epoch+1)/25)","batch_size":32,"time_steps":1,"hard_sigmoid_recurrent_gates":True,"gradient_clip_norm":10,"seed":20261102,"seconds":time.monotonic()-started}
    torch.save({"model":beststate,"config":config,"data_manifest_sha256":sha(DATA/"manifest.json"),"model_source_sha256":sha(ROOT/"milliego/model.py")},path/"best.pt")
    (path/"config.json").write_text(json.dumps(config,indent=2));del model,opt,beststate;torch.cuda.empty_cache()
    return config,path
def transforms(pred):
    T=np.tile(np.eye(4),(len(pred),1,1));T[:,:3,3]=pred[:,:3];T[:,:3,:3]=Rotation.from_euler("xyz",pred[:,3:],degrees=True).as_matrix();return T
def aligned_rmse(pred,gt):
    p=pred[:,:3,3];g=gt[:,:3,3];pc=p-p.mean(0);gc=g-g.mean(0);u,_,vt=np.linalg.svd(pc.T@gc);r=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
    return float(np.sqrt(np.mean(np.sum((pc@r.T-gc)**2,axis=1))))
def metrics(pred,d,ids):
    true=d["labels"][ids];T=transforms(pred);Gt=d["deltas"][ids]
    rerr=Rotation.from_matrix(Gt[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]).magnitude()*180/np.pi
    record={"pairs":len(ids),"translation_vector_rmse_m":float(np.sqrt(np.mean(np.sum((pred[:,:3]-true[:,:3])**2,axis=1)))),"rotation_geodesic_rmse_deg":float(np.sqrt(np.mean(rerr**2)))}
    pairs=d["pairs"][ids];breaks=np.flatnonzero(pairs[1:,0]!=pairs[:-1,1])+1;segments=np.split(np.arange(len(ids)),breaks);trajs=[];records=[]
    for group in segments:
        start=int(pairs[group[0],0]);P=[np.eye(4)]
        for k in group:P.append(P[-1]@T[k])
        P=np.stack(P);frame=np.r_[start,pairs[group,1]]
        actual=np.linalg.inv(d["world_radar"][start])@d["world_radar"][frame]
        err=np.linalg.norm(P[:,:3,3]-actual[:,:3,3],axis=1);length=np.linalg.norm(np.diff(actual[:,:3,3],axis=0),axis=1).sum()
        records.append({"poses":len(P),"path_length_m":float(length),"ate_initial_pose_rmse_m":float(np.sqrt(np.mean(err**2))),"ate_se3_aligned_rmse_m":aligned_rmse(P,actual),"final_position_error_m":float(err[-1])})
        trajs.append((P,actual,d["timestamps"][frame]))
    record["segments"]=records
    return record,trajs
def save_tum(path,T,times):
    q=Rotation.from_matrix(T[:,:3,:3]).as_quat();np.savetxt(path,np.c_[times,T[:,:3,3],q],fmt=["%.9f"]+["%.10f"]*7)
def main():
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=200);p.add_argument("--overfit-steps",type=int,default=400);p.add_argument("--learning-rates",default="1e-5,1e-4");p.add_argument("--output",default="runs/pilot_lab0_gap5_v1");args=p.parse_args()
    output=(ROOT/args.output).resolve();assert ROOT in output.parents;output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available(),"GPU required";torch.set_num_threads(4);torch.manual_seed(20261102)
    started=time.monotonic();checks=invariant_checks();d=dict(np.load(DATA/"labels.npz"));images=np.load(DATA/"images.npy")
    torch.manual_seed(20261102);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt")
    features=cache_features(m,d,images);z=features.cuda();y=torch.from_numpy(d["labels"]).cuda();split=d["split"]
    checks["frozen_cnn_feature_shape"]=list(z.shape);checks["feature_finite"]=bool(torch.isfinite(z).all())
    # Explicitly compare raw radar-input inference with the frozen feature path.
    pair=d["pairs"][0];x=torch.from_numpy(np.concatenate([np.repeat(images[i][None],3,0) for i in pair],0)[None,None]).cuda().float();m.eval()
    with torch.no_grad():error=float((m(x)-m.forward_features(z[:1,None])).abs().max())
    checks["cached_vs_raw_prediction_max_abs_error"]=error;assert error<1e-4
    del m;torch.cuda.empty_cache()
    overfit=fit_overfit(z,y,np.flatnonzero(split==0),args.overfit_steps,output)
    assert overfit["passed_90_percent_reduction"],"Overfit sanity failed: diagnose before full training"
    candidates=[]
    for lr in map(float,args.learning_rates.split(",")):candidates.append(fit(z,y,split,lr,args.epochs,output))
    selected,path=min(candidates,key=lambda item:item[0]["best_val_loss"])
    (output/"selection.json").write_text(json.dumps({"criterion":"lowest validation weighted MSE; no test metrics consulted","candidates":[c for c,_ in candidates],"selected_path":str(path)},indent=2))
    model=RadarOnlyMilliEgo().cuda();checkpoint=torch.load(path/"best.pt",weights_only=True,map_location="cpu");model.load_state_dict(checkpoint["model"]);model.eval()
    all_ids=np.arange(len(split));pred=predict(model,z,all_ids)
    checks["checkpoint_reload_predictions_finite"]=bool(np.isfinite(pred).all())
    with torch.no_grad():
        reloaded=RadarOnlyMilliEgo().cuda();reloaded.load_state_dict(torch.load(path/"best.pt",weights_only=True,map_location="cpu")["model"]);reloaded.eval()
        checks["checkpoint_reload_max_abs_error"]=float((model.forward_features(z[:3,None])-reloaded.forward_features(z[:3,None])).abs().max())
    assert checks["checkpoint_reload_max_abs_error"]==0
    trainmean=d["labels"][split==0].mean(0);results={}
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for s,name in enumerate(["train","val","test"]):
        ids=np.flatnonzero(split==s);results[name]={}
        for method,values in [("model",pred[ids]),("zero_motion",np.zeros_like(pred[ids])),("train_mean",np.tile(trainmean,(len(ids),1)))]:
            met,trajs=metrics(values,d,ids);results[name][method]=met
            if name=="test":
                for j,(P,G,t) in enumerate(trajs):
                    save_tum(output/("test_%s_segment%d.tum"%(method,j)),P,t)
                    if method=="model":save_tum(output/("test_reference_segment%d.tum"%j),G,t)
                    if j==0:
                        ax[0].plot(P[:,0,3],P[:,1,3],label=method)
                        if method=="model":ax[0].plot(G[:,0,3],G[:,1,3],"k--",label="reference")
        np.savez_compressed(output/(name+"_predictions.npz"),pair_rows=ids,prediction=pred[ids],target=d["labels"][ids],pairs=d["pairs"][ids],timestamps=d["timestamps"][d["pairs"][ids]])
    ax[0].set(xlabel="radar-origin x [m]",ylabel="radar-origin y [m]",title="Temporal test segment; initial pose only");ax[0].axis("equal");ax[0].legend();ax[0].grid(alpha=.2)
    for config,folder in candidates:
        history=np.genfromtxt(folder/"history.csv",delimiter=",",names=True)
        ax[1].semilogy(history["epoch"],history["val_loss"],label="val lr=%g"%config["learning_rate"])
    ax[1].set(xlabel="epoch",ylabel="weighted MSE",title="Validation model selection");ax[1].legend();fig.tight_layout();fig.savefig(output/"trajectory_and_loss.png",dpi=170);plt.close(fig)
    record={"scope":"One-sequence temporal holdout pilot, NOT original-paper score or cross-sequence generalization","sequence":"12_21_2020_arpg_lab_run0","selected_config":selected,"metrics":results,"checks":checks,"overfit":overfit,"gpu":torch.cuda.get_device_name(0),"cuda_visible_devices":__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),"torch":torch.__version__,"peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,"wall_seconds":time.monotonic()-started,"data_manifest":str(DATA/"manifest.json"),"model_source_sha256":sha(ROOT/"milliego/model.py"),"training_source_sha256":sha(Path(__file__))}
    (output/"results.json").write_text(json.dumps(record,indent=2));print("COMPLETE",json.dumps({"selected_lr":selected["learning_rate"],"epoch":selected["selected_epoch"],"test":results["test"],"seconds":record["wall_seconds"]}),flush=True)
if __name__=="__main__":main()

