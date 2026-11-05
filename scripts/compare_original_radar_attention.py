"""Original author radar images; one predefined validation fold, fixed attention/no-attention comparison.
Not the author's unpublished standalone executable/checkpoint or a full five-fold paper replication.
"""
import sys,json,time,hashlib,csv,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scipy.spatial.transform import Rotation
from milliego.model import RadarOnlyMilliEgo
from milliego.model_no_attention import RadarOnlyNoAttention
from scripts.validate_projection_diagnosis import metrics,save_tum
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/original_author_radar_v1';OUT=ROOT/'runs/original_radar_control_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loss(p,y):return (p[:,:3]-y[:,:3]).square().mean()+.001*(p[:,3:]-y[:,3:]).square().mean()
def infer(m,z):
    return torch.cat([m.forward_features(chunk[:,None])[:,0].detach().cpu() for chunk in z.split(64)]).numpy()
def prepare():
    out=OUT/'features';out.mkdir(parents=True,exist_ok=True)
    completed=set()
    while True:
        records=[]
        for split in ['train','test']:
            for p in sorted((DATA/split).glob('*.json')) if (DATA/split).exists() else []:
                rec=json.loads(p.read_text());name=rec['sequence'];dest=out/(name+'.pt')
                if name not in completed:
                    if not dest.exists():
                        raw=np.load(DATA/split/(name+'.npz'));images=raw['images'][:,0,:,:,0];labels=raw['labels'];times=raw['timestamps']
                        assert len(images)==len(labels)+1
                        z=[]
                        with torch.no_grad():
                            for first in range(0,len(labels),16):
                                a=torch.from_numpy(images[first:first+16].copy()).cuda()
                                b=torch.from_numpy(images[first+1:first+17].copy()).cuda()
                                # Last batch must have exactly equal endpoint counts.
                                a=a[:len(b)];x=torch.cat([a[:,None].repeat(1,3,1,1),b[:,None].repeat(1,3,1,1)],1)
                                z.append(cnn.cnn(x).cpu())
                        features=torch.cat(z);assert len(features)==len(labels) and torch.isfinite(features).all()
                        torch.save({'features':features,'labels':torch.from_numpy(labels.astype(np.float32)),'timestamps':torch.from_numpy(times),'source_npz_sha256':rec['npz_sha256'],'split':split},dest)
                    completed.add(name);print('FEATURES',name,flush=True)
                records.append((name,split,dest))
        if len(records)==57 and (DATA/'manifest.json').exists():
            return records
        # Polling is inside the detached background worker, not a tool/UI wait.
        time.sleep(5)
def dataset(record):
    name,split,path=record;d=torch.load(path,weights_only=True,map_location='cpu');return name,d
def make_reference(labels,timestamps):
    n=len(labels);D=np.tile(np.eye(4),(n,1,1));D[:,:3,3]=labels[:,:3];D[:,:3,:3]=Rotation.from_euler('XYZ',labels[:,3:],degrees=True).as_matrix()
    W=[np.eye(4)]
    for t in D:W.append(W[-1]@t)
    return dict(labels=labels,deltas=D,pairs=np.c_[np.arange(n),np.arange(1,n+1)],world_radar=np.stack(W),timestamps=timestamps.astype(np.float64)/1e9)
if __name__=='__main__':
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    torch.manual_seed(20261102);cnn=RadarOnlyMilliEgo().cuda();cnn.load_cnn(ROOT/'weights/cnn_torch.pt');cnn.eval()
    records=prepare();del cnn;torch.cuda.empty_cache()
    trainrecords=sorted([r for r in records if r[1]=='train'],key=lambda r:int(r[0].split('_')[-1]))
    valnames={r[0] for i,r in enumerate(trainrecords) if i%5==0}
    train=[dataset(r) for r in trainrecords if r[0] not in valnames];val=[dataset(r) for r in trainrecords if r[0] in valnames]
    vz=torch.cat([d['features'] for _,d in val]).cuda();vy=torch.cat([d['labels'] for _,d in val]).cuda()
    train=[(name,{**d,'features':d['features'].cuda(),'labels':d['labels'].cuda()}) for name,d in train]
    cfg={'arms':['without_attention','with_attention'],'seed':20261102,'epochs':201,'learning_rate':1e-5,'optimizer':'RMSprop alpha=.9 eps=1e-8','cnn':'official pretrained and frozen','sampling':'Official train_cross_att.py: shuffled sequence order, random starting positions, contiguous 64-pair batches, floor((N-66)/64) batches/sequence/epoch','normalization':'Use author HDF5 values unchanged; no second normalization','label_order':'tx ty tz + intrinsic XYZ degrees, verified against original euler2mat','validation':'One predefined fold, every fifth numerically sorted author training sequence (index modulo5=0). All eight provided author test sequences untouched during training.','train_sequences':[n for n,_ in train],'validation_sequences':sorted(valnames),'test_sequences':[r[0] for r in records if r[1]=='test'],'selection':'lowest validation loss; never select on test','limitations':['Reconstructed standalone PyTorch model, using only radar without IMU or attention; public upstream exposes fusion builder only.','One validation fold/seed, not full five-fold paper reproduction.','Provided test folder has eight sequences; paper standalone figure uses three selected sequences; not the same test summary.'],'model_sources':{f:sha(ROOT/f) for f in ['milliego/model.py','milliego/model_no_attention.py']}}
    (OUT/'protocol.json').write_text(json.dumps(cfg,indent=2));print('PROTOCOL',json.dumps(cfg),flush=True)
    starts={};sharedhashes=[]
    for arm in cfg['arms']:
        folder=OUT/arm;folder.mkdir(exist_ok=True)
        if (folder/'training_complete.json').exists():
            previous=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
            previous_protocol=dict(previous['protocol'])
            previous_protocol['arms']=cfg['arms']
            assert previous_protocol==cfg
            sharedhashes.append(previous['shared_initial_sha256'])
            continue
        seed=cfg['seed'];torch.manual_seed(seed);rng=np.random.RandomState(seed)
        m=(RadarOnlyMilliEgo if arm=='with_attention' else RadarOnlyNoAttention)().cuda();m.load_cnn(ROOT/'weights/cnn_torch.pt')
        digest=hashlib.sha256()
        for k,v in m.state_dict().items():
            if not k.startswith('mask.'):digest.update(k.encode());digest.update(v.detach().cpu().numpy().tobytes())
        sharedhashes.append(digest.hexdigest());assert len(set(sharedhashes))==1
        opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-5,alpha=.9,eps=1e-8)
        best=float('inf');history=[];step=0;t0=time.monotonic()
        for epoch in range(cfg['epochs']):
            m.train();order=rng.permutation(len(train));total=0;count=0
            for i in order:
                name,d=train[i];n=len(d['labels']);indices=np.arange(max(0,n-65));rng.shuffle(indices)
                for j in range(len(indices)//64):
                    start=int(indices[j*64]);z=d['features'][start:start+64];y=d['labels'][start:start+64]
                    assert len(z)==64
                    opt.zero_grad(set_to_none=True);pred=m.forward_features(z[:,None])[:,0];L=loss(pred,y);assert torch.isfinite(L);L.backward();opt.step();step+=1;total+=float(L.detach())*64;count+=64
            m.eval()
            with torch.no_grad():
                vl=0
                for a in range(0,len(vy),64):
                    pp=m.forward_features(vz[a:a+64,None])[:,0];vl+=float(loss(pp,vy[a:a+64]))*len(pp)
                vl/=len(vy)
            history.append({'epoch':epoch,'updates':step,'train_dropout_loss':total/count,'validation_loss':vl})
            if vl<best:
                best=vl;torch.save({'model':m.state_dict(),'arm':arm,'protocol':cfg,'epoch':epoch,'updates':step,'validation_loss':vl,'shared_initial_sha256':digest.hexdigest()},folder/'best.pt')
            if epoch%10==0 or epoch==200:print('TRAIN',arm,epoch,step,total/count,vl,'seconds',time.monotonic()-t0,flush=True)
        with (folder/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
        (folder/'training_complete.json').write_text(json.dumps({'updates':step,'best_validation_loss':best,'epochs':201,'seconds':time.monotonic()-t0},indent=2))
        del m,opt;torch.cuda.empty_cache()
    # Test every provided test sequence only after both arms finish.
    results=[]
    for arm in cfg['arms']:
        folder=OUT/arm;cp=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
        m=(RadarOnlyMilliEgo if arm=='with_attention' else RadarOnlyNoAttention)().cuda();m.load_state_dict(cp['model']);m.eval()
        for rec in sorted([r for r in records if r[1]=='test']):
            name,d=dataset(rec);z=d['features'].cuda()
            with torch.no_grad():p=infer(m,z)
            labels=d['labels'].numpy();times=d['timestamps'].numpy();ref=make_reference(labels,times);met,trajs=metrics(p,ref,np.arange(len(p)),'XYZ')
            P,G,t=trajs[0];pp=folder/(name+'_prediction.tum');gg=folder/(name+'_reference.tum')
            save_tum(pp,P[1:],t[1:]);save_tum(gg,G[1:],t[1:])
            # The upstream std_ate implementation compares positions directly, without alignment.
            original_ate=float(np.sqrt(np.mean(np.sum((P[1:,:3,3]-G[1:,:3,3])**2,axis=1))))
            met['ate_source_convention_rmse_m']=original_ate
            met['path_length_m']=float(np.linalg.norm(np.diff(G[:,:3,3],axis=0),axis=1).sum())
            met['ate_over_path_percent']=100*original_ate/met['path_length_m'] if met['path_length_m'] else None
            np.savez_compressed(folder/(name+'_prediction.npz'),prediction=p,labels=labels,timestamps=times)
            results.append({'arm':arm,'sequence':name,'selected_epoch':cp['epoch'],'metrics':met})
            print('RESULT',arm,name,json.dumps(met),flush=True)
            del z
        del m;torch.cuda.empty_cache()
    aggregate={arm:{k:float(np.mean([r['metrics'][k] for r in results if r['arm']==arm])) for k in ['ate_source_convention_rmse_m','ate_initial_rmse_m','ate_se3_rmse_m','pair_translation_rmse_m','pair_rotation_rmse_deg','ate_over_path_percent']} for arm in cfg['arms']}
    (OUT/'results.json').write_text(json.dumps({'protocol':cfg,'test_per_sequence':results,'test_macro_mean':aggregate},indent=2))
    print('COMPLETE',json.dumps(aggregate),flush=True)

