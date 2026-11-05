"""Fixed-algorithm expanded-data vs repeated-base control, same update counts and physical evaluation."""
import json,sys,time,math,csv,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from milliego.model import RadarOnlyMilliEgo
from scripts.validate_native_sampling import compose_to_common
from scripts.validate_projection_diagnosis import metrics,save_tum
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/coloradar_lab_expanded_v1";BASE=ROOT/"data/coloradar_lab0_native10_v1";OLD=ROOT/"data/coloradar_lab0_contractfix_v2";OUT=ROOT/"runs/direct_halfsecond_v1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loss(p,y):return (p[:,:3]-y[:,:3]).square().mean()+.001*(p[:,3:]-y[:,3:]).square().mean()
def load_features(data,rawfile,mean,cache):
    provenance={"images_sha256":sha(rawfile),"labels_sha256":sha(data/"labels.npz"),"mean":mean,"cnn_sha256":sha(ROOT/"weights/cnn_torch.pt"),"model_source_sha256":sha(ROOT/"milliego/model.py")}
    if cache.exists():
        saved=torch.load(cache,weights_only=True,map_location="cpu");assert saved["provenance"]==provenance
        return saved["features"]
    images=np.load(rawfile,mmap_mode="r");d=np.load(data/"labels.npz");pairs=d["pairs"]
    torch.manual_seed(20261102);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt");m.eval();out=[]
    with torch.no_grad():
        for j in range(0,len(pairs),16):
            p=pairs[j:j+16];a=torch.from_numpy(np.array(images[p[:,0]],copy=True)).cuda().float()/255-mean;b=torch.from_numpy(np.array(images[p[:,1]],copy=True)).cuda().float()/255-mean
            x=torch.cat([a[:,None].repeat(1,3,1,1),b[:,None].repeat(1,3,1,1)],1);z=m.cnn(x);assert torch.isfinite(z).all();out.append(z.cpu())
    features=torch.cat(out);torch.save({"features":features,"provenance":provenance},cache);del m
    print("FEATURES",str(data),list(features.shape),flush=True);return features
def main():
    OUT.mkdir(exist_ok=True);manifest=json.loads((DATA/"manifest.json").read_text());mean=manifest["common_training_mean"];base=dict(np.load(OLD/"labels.npz"));old=dict(np.load(OLD/"labels.npz"));base_train=np.flatnonzero(base["split"]==0);val=np.flatnonzero(base["split"]==1)
    OUT.joinpath("features").mkdir(exist_ok=True)
    bz=load_features(OLD,ROOT/"data/coloradar_lab0_gap5_v1/images.npy",mean,OUT/"features/base.pt")
    zz=[bz[base_train]];yy=[base["labels"][base_train]]
    for rec in manifest["sequences"]:
        p=OUT/rec["sequence"];zz.append(load_features(p,DATA/rec["sequence"]/"images_raw.npy",mean,OUT/"features"/(rec["sequence"]+".pt")));yy.append(np.load(p/"labels.npz")["labels"])
    trainz=torch.cat(zz).cuda();trainy=torch.from_numpy(np.concatenate(yy)).cuda();basez=bz.cuda();basey=torch.from_numpy(base["labels"]).cuda()
    del zz,yy,bz
    n=len(trainy);base_n=len(base_train);batches=math.ceil(5593/64);val=torch.as_tensor(val,device="cuda")
    cfg={"seeds":[20261102,20261103,20261104],"arms":["expanded_data"],"epochs":200,"batch_size":64,"learning_rate":1e-5,"schedule":"constant","optimizer":"RMSprop alpha=.9 eps=1e-8","loss":"MSE translation m + .001 MSE rotation degrees","cnn_frozen":True,"time_steps":1,"gradient_clip":None,"base_train_pairs":base_n,"expanded_train_pairs":n,"steps_per_arm":batches*200,"small_update_budget":2200,"normalization_mean_same_in_all_arms":mean,"evaluation":"Same 46 run0 test intervals, directly predicted at 0.5 s","epoch_selection":"validation loss only at fixed checkpoints; no test-based hyperparameter selection","scope":"Direct 0.5 s radar pairs with same 17600 updates; existing run0 development evaluation","model_source_sha256":sha(ROOT/"milliego/model.py"),"data_manifest_sha256":sha(DATA/"manifest.json")}
    (OUT/"protocol.json").write_text(json.dumps(cfg,indent=2));print("PROTOCOL",json.dumps(cfg),flush=True)
    start=time.monotonic();pending=[]
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            folder=OUT/(arm+"_seed%d"%seed);folder.mkdir(exist_ok=True)
            if (folder/"training_complete.json").exists():
                done=json.loads((folder/"training_complete.json").read_text());pending.extend((folder,arm,seed,mode) for mode in done["modes"]);print("REUSE",folder,flush=True);continue
            torch.manual_seed(seed);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt");opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-5,alpha=.9,eps=1e-8)
            initial=hashlib.sha256()
            for k,v in m.state_dict().items():initial.update(k.encode());initial.update(v.detach().cpu().numpy().tobytes())
            initial_hash=initial.hexdigest();history=[];best=float("inf");step=0;saved={}
            for epoch in range(200):
                m.train()
                if arm=="expanded_data":order=torch.cat((torch.randperm(n,device="cuda"),torch.randperm(n,device="cuda")[:5593-n]))
                else:
                    order=torch.arange(base_n,device="cuda").repeat(math.ceil(n/base_n))[:n]
                    order=order[torch.randperm(n,device="cuda")]
                X=trainz if arm=="expanded_data" else trainz[:base_n];Y=trainy if arm=="expanded_data" else trainy[:base_n]
                total=0;count=0
                for batch in order.split(64):
                    opt.zero_grad(set_to_none=True);p=m.forward_features(X[batch,None])[:,0];L=loss(p,Y[batch]);assert torch.isfinite(L)
                    L.backward()
                    if step==0:assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                    opt.step();step+=1;total+=float(L.detach())*len(batch);count+=len(batch)
                    if step==cfg["small_update_budget"]:
                        m.eval()
                        with torch.no_grad():pv=m.forward_features(basez[val,None])[:,0];vl=float(loss(pv,basey[val]))
                        if vl<best:best=vl;be=epoch+1;bs=step;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                        saved["first_2200_updates"]=(state,be,bs,best);m.train()
                m.eval()
                with torch.no_grad():
                    p=m.forward_features(basez[val,None])[:,0];vl=float(loss(p,basey[val]));trans=float((p[:,:3]-basey[val,:3]).square().mean());rot=float((p[:,3:]-basey[val,3:]).square().mean())
                history.append({"epoch":epoch+1,"updates":step,"train_dropout_loss":total/count,"val_loss":vl,"val_translation_mse":trans,"val_rotation_degree_mse":rot})
                if vl<best:best=vl;be=epoch+1;bs=step;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                if epoch%10==0 or epoch==199:print("TRAIN",arm,seed,epoch+1,"steps",step,"train",total/count,"val",vl,flush=True)
            saved["full_200_epochs"]=(state,be,bs,best)
            with (folder/"history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
            for mode,(state,be,bs,best) in saved.items():
                torch.save({"model":state,"config":cfg,"selected_epoch":be,"selected_updates":bs,"selected_val_loss":best,"seed":seed,"arm":arm,"initial_state_sha256":initial_hash,"normalization_mean":mean,"euler_order":"XYZ"},folder/(mode+".pt"));pending.append((folder,arm,seed,mode))
            (folder/"training_complete.json").write_text(json.dumps({"modes":list(saved),"initial_state_sha256":initial_hash,"updates":step,"training_complete":True},indent=2));del m,opt;torch.cuda.empty_cache()
    # No evaluation on held-out run0 segment until all fixed training arms and seeds are complete.
    rows=[];checks=[]
    for folder,arm,seed,mode in pending:
        cp=torch.load(folder/(mode+".pt"),weights_only=True,map_location="cpu");m=RadarOnlyMilliEgo().cuda();m.load_state_dict(cp["model"]);m.eval()
        with torch.no_grad():p=torch.cat([m.forward_features(basez[batch,None])[:,0].cpu() for batch in torch.arange(len(basez),device="cuda").split(64)]).numpy()
        rec={"arm":arm,"seed":seed,"mode":mode,"selected_epoch":cp["selected_epoch"],"selected_updates":cp["selected_updates"],"selected_val_loss":cp["selected_val_loss"],"initial_state_sha256":cp["initial_state_sha256"]}
        for s,name in enumerate(["train","val","test"]):
            ids=np.flatnonzero(old["split"]==s);pred=p[ids];met,trajs=metrics(pred,old,ids,"XYZ");rec[name]=met
            if name=="test":
                np.savez_compressed(folder/(mode+"_predictions.npz"),prediction=pred,native_prediction=p,pair_rows=ids)
                for j,(P,G,t) in enumerate(trajs):
                    target=folder/(mode+"_test%d.tum"%j);reference=folder/(mode+"_reference%d.tum"%j);save_tum(target,P,t);save_tum(reference,G,t)
                    pp=np.loadtxt(target);gg=np.loadtxt(reference);a=float(np.sqrt(np.mean(np.sum((pp[:,1:4]-gg[:,1:4])**2,1))));assert abs(a-met["segments"][j]["ate_initial_rmse_m"])<1e-8
        # Check raw radar-image inference against cached pipeline at same physical test pair.
        if mode=="full_200_epochs":
            i=int(np.flatnonzero(base["split"]==2)[0]);pair=base["pairs"][i];raw=np.load(ROOT/"data/coloradar_lab0_gap5_v1/images.npy",mmap_mode="r")[pair].astype(np.float32)/255-np.float32(mean)
            x=torch.from_numpy(np.concatenate([np.repeat(im[None],3,0) for im in raw],0)[None,None]).cuda()
            with torch.no_grad():direct=m(x)[0,0].cpu().numpy();cached=m.forward_features(basez[i:i+1,None])[0,0].cpu().numpy()
            error=float(abs(direct-cached).max());assert error<1e-3
            checks.append({"arm":arm,"seed":seed,"raw_vs_cached_max_abs_error":error})
        rows.append(rec);print("RESULT",arm,seed,mode,json.dumps(rec["test"]),flush=True);del m
    agg={}
    for arm in cfg["arms"]:
        agg[arm]={}
        for mode in ["first_2200_updates","full_200_epochs"]:
            group=[r for r in rows if r["arm"]==arm and r["mode"]==mode]
            agg[arm][mode]={k:{"mean":float(np.mean([r["test"][k] for r in group])),"std":float(np.std([r["test"][k] for r in group])),"values":[r["test"][k] for r in group]} for k in ["pair_translation_rmse_m","pair_rotation_rmse_deg","ate_initial_rmse_m","ate_se3_rmse_m","endpoint_error_m"]}
    for seed in cfg["seeds"]:assert len({r["initial_state_sha256"] for r in rows if r["seed"]==seed})==1
    report={"protocol":cfg,"runs":rows,"test_summary":agg,"verification":checks,"model_unchanged":sha(ROOT/"milliego/model.py")==cfg["model_source_sha256"],"seconds":time.monotonic()-start}
    assert report["model_unchanged"];(OUT/"results.json").write_text(json.dumps(report,indent=2));print("COMPLETE",json.dumps(agg),flush=True)
if __name__=="__main__":main()

