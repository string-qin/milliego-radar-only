"""GPU pipeline diagnostic, NOT a completed/equivalence-validated milliEgo port."""
import json,os,time,platform,subprocess,io
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
class SameConv(nn.Module):
    def __init__(self,ci,co,k,s):
        super().__init__();self.conv=nn.Conv2d(ci,co,k,stride=s,padding=0);self.k=k;self.s=s
    def forward(self,x):
        h,w=x.shape[-2:];ph=max(0,((h+self.s-1)//self.s-1)*self.s+self.k-h);pw=max(0,((w+self.s-1)//self.s-1)*self.s+self.k-w)
        return F.leaky_relu(self.conv(F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2))),negative_slope=.1)
class RadarOnlyDiagnostic(nn.Module):
    def __init__(self):
        super().__init__();dims=[(6,64,7,2),(64,128,5,2),(128,256,5,2),(256,256,3,1),(256,512,3,2),(512,512,3,1),(512,512,3,2),(512,512,3,1),(512,1024,3,2)]
        self.cnn=nn.Sequential(*(SameConv(*d) for d in dims))
        self.mask=nn.Sequential(nn.Linear(1024,4,bias=False),nn.ReLU(),nn.Linear(4,1024,bias=False),nn.Sigmoid())
        self.input_dropout=nn.Dropout(.25)
        self.lstm1=nn.LSTM(4096,512,batch_first=True);self.lstm2=nn.LSTM(512,512,batch_first=True)
        def head():return nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(.25),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
        self.translation=head();self.rotation=head()
    def forward(self,x):
        b,t,c,h,w=x.shape;z=self.cnn(x.reshape(b*t,c,h,w));m=self.mask(z.mean((-2,-1)))[:,:,None,None];z=z*m
        z=z.permute(0,2,3,1).reshape(b,t,4096)
        z,_=self.lstm1(self.input_dropout(z));z,_=self.lstm2(z)
        return torch.cat((self.translation(z),self.rotation(z)),dim=-1)
def main():
    assert torch.cuda.is_available(),"CUDA unavailable: no silent CPU fallback"
    torch.set_num_threads(4);torch.manual_seed(20261102);np.random.seed(20261102)
    report={"python":platform.python_version(),"torch":torch.__version__,"torch_cuda_runtime":torch.version.cuda,
      "cudnn":torch.backends.cudnn.version(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
      "gpu":torch.cuda.get_device_name(0),"compute_capability":list(torch.cuda.get_device_capability(0)),
      "torch_arch_list":torch.cuda.get_arch_list(),"scope":"Real radar input; synthetic zero regression targets. A short GPU compute test, not training accuracy or a completed faithful port.",
      "diagnostic_model_notes":["9 CNN layers and channel mask derived from official networks.py","two 512-unit LSTMs and separate 128/64/3 heads","TF SAME padding implemented explicitly","IMU and cross-modal branches removed","random initialization; no cnn.h5 conversion or feature freezing","input normalized to [0,1] for diagnostic only","dropout approximation differs from Keras dropout_W","original default sequence dimension is 1; also exercising T=3"]}
    start=time.monotonic();a=torch.randn(512,512,device="cuda",requires_grad=True);(a@a.T).square().mean().backward();torch.cuda.synchronize()
    report["cuda_matmul_backward_finite"]=bool(torch.isfinite(a.grad).all());del a
    data=np.load(ROOT/"reports/coloradar_diagnostic_inputs.npz")["images"]
    pairs=np.stack([np.concatenate((np.repeat(data[i][None],3,0),np.repeat(data[i+1][None],3,0)),axis=0) for i in range(3)])
    x=torch.from_numpy(pairs[None]).float().cuda()
    model=RadarOnlyDiagnostic().cuda();opt=torch.optim.RMSprop(model.parameters(),lr=1e-5,alpha=.9,eps=1e-8)
    before=model.translation[-1].weight.detach().clone()
    losses=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True);y=model(x)
        loss=y[...,:3].square().mean()+.001*y[...,3:].square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step();losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    report.update({"input_shape":list(x.shape),"output_shape":list(y.shape),"parameter_count":sum(p.numel() for p in model.parameters()),
      "optimizer_steps":3,"finite_gradients":True,"synthetic_target_losses":losses,"parameters_changed":bool(not torch.equal(before,model.translation[-1].weight)),
      "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20})
    model.eval()
    with torch.no_grad():expected=model(x).cpu();single=model(x[:,:1])
    report["single_pair_output_shape"]=list(single.shape)
    checkpoint=ROOT/"reports/diagnostic_gpu_state.pt";torch.save({"model":model.state_dict(),"scope":"random diagnostic only"},checkpoint)
    restored=RadarOnlyDiagnostic().cuda();restored.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cuda")["model"]);restored.eval()
    with torch.no_grad():actual=restored(x).cpu()
    report["checkpoint_reload_max_abs_error"]=float((expected-actual).abs().max())
    assert torch.equal(expected,actual)
    # DataLoader workers and HDF5 sample round-trip are part of the actual installed environment check.
    from torch.utils.data import DataLoader,TensorDataset
    loader=DataLoader(TensorDataset(torch.from_numpy(pairs)),batch_size=1,num_workers=2)
    report["dataloader_batches"]=sum(1 for _ in loader)
    import h5py,scipy,yaml,matplotlib,PIL,tensorboard
    with h5py.File(ROOT/"reports/diagnostic_inputs.h5","w") as f:f.create_dataset("radar_pairs",data=pairs)
    with h5py.File(ROOT/"reports/diagnostic_inputs.h5","r") as f:assert np.array_equal(f["radar_pairs"][:],pairs)
    report["dependency_imports_and_hdf5_roundtrip"]=True
    report["wall_seconds"]=time.monotonic()-start
    (ROOT/"reports/gpu_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()

