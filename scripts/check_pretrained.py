import json,hashlib
from pathlib import Path
import numpy as np,h5py,torch
from check_gpu import RadarOnlyDiagnostic
ROOT=Path(__file__).resolve().parents[1]
names=["conv1","conv2","conv3","conv3_1","conv4","conv4_1","conv5","conv5_1","conv6"]
model=RadarOnlyDiagnostic()
record={"scope":"HDF5 tensor conversion + independent NumPy conv1 check, NOT a full TensorFlow/PyTorch model parity test.","layers":[]}
state={}
with h5py.File(ROOT/"weights/cnn.h5","r") as f:
    for i,name in enumerate(names):
        g=f["model_weights"][name];w=np.array(g[name+"_W:0"]);b=np.array(g[name+"_b:0"])
        converted=np.transpose(w,(3,2,0,1)).copy()
        assert tuple(converted.shape)==tuple(model.cnn[i].conv.weight.shape)
        assert np.isfinite(w).all() and np.isfinite(b).all()
        assert np.array_equal(np.transpose(converted,(2,3,1,0)),w)
        model.cnn[i].conv.weight.data.copy_(torch.from_numpy(converted));model.cnn[i].conv.bias.data.copy_(torch.from_numpy(b))
        state[str(i)+".conv.weight"]=torch.from_numpy(converted);state[str(i)+".conv.bias"]=torch.from_numpy(b)
        record["layers"].append({"name":name,"source_shape":list(w.shape),"torch_shape":list(converted.shape),"exact_tensor_roundtrip":True})
    record["intentionally_unused_layers"]=["conv6_1","fc_pose"]
    w=np.array(f["model_weights/conv1/conv1_W:0"],dtype=np.float64);b=np.array(f["model_weights/conv1/conv1_b:0"],dtype=np.float64)
rng=np.random.default_rng(2);x=rng.normal(size=(1,9,13,6)).astype(np.float32)
pad=np.pad(x.astype(float),((0,0),(3,3),(3,3),(0,0)))
windows=np.lib.stride_tricks.sliding_window_view(pad,(7,7),axis=(1,2))[:,::2,::2,:,:,:]
ref=np.einsum("nhwckl,klco->nhwo",windows,w)+b;ref=np.where(ref>=0,ref,.1*ref)
with torch.no_grad():actual=model.cnn[0](torch.from_numpy(x.transpose(0,3,1,2))).permute(0,2,3,1).numpy()
error=float(np.max(abs(actual-ref)));record["conv1_numpy_reference_max_abs_error"]=error
assert np.allclose(actual,ref,atol=1e-5,rtol=1e-4)
torch.save({"cnn_state_dict":state,"source_sha256":hashlib.sha256((ROOT/"weights/cnn.h5").read_bytes()).hexdigest(),"layers":names},ROOT/"weights/cnn_torch.pt")
assert torch.cuda.is_available()
model=model.cuda().eval();images=np.load(ROOT/"reports/coloradar_diagnostic_inputs.npz")["images"]
pair=np.concatenate([np.repeat(images[i][None],3,0) for i in (0,1)],axis=0)
with torch.no_grad():y=model(torch.from_numpy(pair[None,None]).cuda())
record["pretrained_cnn_full_diagnostic_gpu_forward_finite"]=bool(torch.isfinite(y).all())
assert record["pretrained_cnn_full_diagnostic_gpu_forward_finite"]
record["converted_path"]=str(ROOT/"weights/cnn_torch.pt")
(ROOT/"reports/pretrained_conversion.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))

