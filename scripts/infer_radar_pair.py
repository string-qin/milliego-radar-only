"""Infer one relative pose using two radar point clouds and a checkpoint; no ground truth or IMU inputs."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from milliego.model import RadarOnlyMilliEgo
from prepare_single_sequence import project
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--first",required=True);p.add_argument("--second",required=True);p.add_argument("--output");args=p.parse_args()
    assert torch.cuda.is_available(),"GPU required";torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model=RadarOnlyMilliEgo().cuda();model.load_state_dict(checkpoint["model"],strict=True);model.eval()
    ims=[]
    for file in [args.first,args.second]:
        points=np.fromfile(file,dtype="<f4").reshape(-1,5);image,_,_=project(points);ims.append(image)
    x=np.concatenate([np.repeat(im[None],3,0) for im in ims],axis=0)[None,None]
    with torch.no_grad():value=model(torch.from_numpy(x).cuda().float())[0,0].cpu().numpy()
    assert np.isfinite(value).all()
    record={"translation_m":value[:3].tolist(),"rotation_xyz_degrees":value[3:].tolist(),"transform":"T_radar_first_from_radar_second = inv(T_world_radar_first) @ T_world_radar_second","inputs":[args.first,args.second],"uses_imu":False,"uses_ground_truth":False,"checkpoint":args.checkpoint}
    text=json.dumps(record,indent=2)
    if args.output:Path(args.output).write_text(text)
    print(text)
if __name__=="__main__":main()

