"""Training/validation diagnostic only: never evaluate test labels or metrics."""
import sys,json,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scipy.spatial.transform import Rotation
from milliego.model import RadarOnlyMilliEgo
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/diagnosis_lab0_v1";OUT.mkdir(exist_ok=False)
torch.set_num_threads(4);assert torch.cuda.is_available()
d=np.load(ROOT/"data/coloradar_lab0_gap5_v1/labels.npz")
train_ids=np.flatnonzero(d["split"]==0);val_ids=np.flatnonzero(d["split"]==1)
f=torch.load(ROOT/"data/coloradar_lab0_gap5_v1/frozen_cnn_features.pt",weights_only=True)["features"]
# Only training and validation samples are moved into this diagnostic.
z=torch.cat([f[train_ids],f[val_ids]]).cuda()
target=torch.from_numpy(np.concatenate([d["labels"][train_ids],d["labels"][val_ids]])).cuda()
n=len(train_ids);tr=torch.arange(n,device="cuda");va=torch.arange(n,len(z),device="cuda")
def loss(pred,y):return (pred[:,:3]-y[:,:3]).square().mean()+.001*(pred[:,3:]-y[:,3:]).square().mean()
def metric(p,y):
    p=p.detach().cpu().numpy();y=y.detach().cpu().numpy()
    rel=Rotation.from_euler("xyz",y[:,3:],degrees=True).inv()*Rotation.from_euler("xyz",p[:,3:],degrees=True)
    return {"translation_rmse_m":float(np.sqrt(np.mean(np.sum((p[:,:3]-y[:,:3])**2,1)))),"rotation_rmse_deg":float(np.sqrt(np.mean(np.degrees(rel.magnitude())**2))),"rotation_prediction_std_deg":p[:,3:].std(0).tolist(),"weighted_mse":float(np.mean((p[:,:3]-y[:,:3])**2)+.001*np.mean((p[:,3:]-y[:,3:])**2))}
report={"scope":"Diagnosis of existing training and validation only; no test model selection or evaluation.","experiments":[]}
mean=target[tr].mean(0)
report["baselines"]={name:{"zero":metric(torch.zeros_like(target[ids]),target[ids]),"train_mean":metric(mean.expand(len(ids),-1),target[ids])} for name,ids in [("train",tr),("validation",va)]}
for name,lr,scale,drop in [
("constant_lr1e-5",1e-5,1.,True),
("constant_lr1e-4",1e-4,1.,True),
("constant_lr1e-3",1e-3,1.,True),
("rotation_output_scale10_lr1e-4",1e-4,10.,True),
("no_dropout_lr1e-3",1e-3,1.,False)]:
    torch.manual_seed(20261102);m=RadarOnlyMilliEgo().cuda();m.load_cnn(ROOT/"weights/cnn_torch.pt")
    opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=lr,alpha=.9,eps=1e-8)
    units=torch.tensor([1,1,1,scale,scale,scale],device="cuda");best=math.inf;beststep=0;hist=[];initial={};start=time.monotonic()
    def forward(ids):return m.forward_features(z[ids,None])[:,0]*units
    m.eval()
    with torch.no_grad():initial={"train":metric(forward(tr),target[tr]),"validation":metric(forward(va),target[va])}
    gradient=None
    for step in range(1000):
        m.train(drop);ids=tr[torch.randperm(len(tr),device="cuda")[:32]]
        opt.zero_grad(set_to_none=True);p=forward(ids);L=loss(p,target[ids]);assert torch.isfinite(L);L.backward()
        if step==0:
            gradient={k:float(getattr(m,k)[-1].weight.grad.norm()) for k in ["translation","rotation"]}
        torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
        if step in [0,24,99,249,499,749,999]:
            m.eval()
            with torch.no_grad():met={"step":step+1,"train":metric(forward(tr),target[tr]),"validation":metric(forward(va),target[va])}
            hist.append(met)
            if met["validation"]["weighted_mse"]<best:
                best=met["validation"]["weighted_mse"];beststep=step+1;bestmetrics=met
            print(name,step+1,"train",met["train"]["weighted_mse"],"val",met["validation"]["weighted_mse"],"rot_train",met["train"]["rotation_rmse_deg"],flush=True)
    rec={"name":name,"lr":lr,"rotation_output_scale":scale,"dropout_enabled":drop,"steps":1000,"initial":initial,"initial_head_grad_norm":gradient,"best_validation_step":beststep,"best_validation_metrics":bestmetrics,"final":hist[-1],"history":hist,"seconds":time.monotonic()-start}
    report["experiments"].append(rec);(OUT/"results.json").write_text(json.dumps(report,indent=2));del m,opt;torch.cuda.empty_cache()
# Can frozen features linearly predict labels? Ridge diagnostic with train-only centering and fixed lambda.
X=f[train_ids].flatten(1).numpy().astype(float);V=f[val_ids].flatten(1).numpy().astype(float);Y=d["labels"][train_ids].astype(float);Yv=d["labels"][val_ids]
center=X.mean(0);X-=center;V-=center;factor=np.sqrt(np.mean(X**2));X/=factor;V/=factor
K=X@X.T/X.shape[1];Kv=V@X.T/X.shape[1];ym=Y.mean(0)
ridge=[]
for reg in [.0001,.01,1.]:
    coef=np.linalg.solve(K+reg*np.eye(len(X)),Y-ym);pt=K@coef+ym;pv=Kv@coef+ym
    ridge.append({"lambda":reg,"train":metric(torch.tensor(pt),torch.tensor(Y)),"validation":metric(torch.tensor(pv),torch.tensor(Yv))})
report["linear_feature_probe"]=ridge
report["input_feature_stats"]={"finite":bool(torch.isfinite(z).all()),"abs_mean":float(z.abs().mean()),"std":float(z.std())}
report["notes"]=["Constant LR matters: upstream calls model.fit(epochs=1) repeatedly and passes LearningRateScheduler only in last inner call; its callback epoch is reset each fit, so the posted step-decay callback does not implement a normal 200-epoch schedule.","The pilot had 138 training pairs and 5 batches/epoch; 200 epochs =1000 optimizer steps. Pilot decay reduced LR to ~0.1 initial by the end.","Learning-rate/output-scale experiments are adaptation diagnostics, not original-paper parity or independent test improvement."]
(OUT/"results.json").write_text(json.dumps(report,indent=2));print("COMPLETE",flush=True)

