import sys,json,importlib.util,hashlib,inspect
from pathlib import Path
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
src=ROOT/'upstream/utility/pcl2depth.py'
a=mod(src,'original');b=mod(ROOT/'scripts/prepare_single_sequence.py','current')
cfg=dict(v_res=float(np.nextafter(60/63,np.inf)),h_res=float(np.nextafter(120/255,np.inf)),v_fov=(-30,30),h_fov=(-60,60),max_v=30,depth=True)
out={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'fixture_parameters_not_author_configuration':cfg,'synthetic':[]}
for name,pts in [('same_pixel_order1',[[5,0,0],[10,0,0]]),('same_pixel_order2',[[10,0,0],[5,0,0]]),('elevated_point',[[3,4,2]]),('empty_fov',[[-1,0,0]])]:
 p=np.array(pts,dtype=float);x=a.velo_points_2_pano(p,**cfg);y=b.project(p)[0]
 out['synthetic'].append(dict(name=name,original_shape=list(x.shape),current_shape=list(y.shape),original_nonzero=x[x>0].tolist(),current_nonzero=y[y>0].tolist(),same=bool(np.array_equal(x,y))))
pcfiles=sorted(Path('/storage/data/metaiot_data/public_dataset/coloradar/12_21_2020_arpg_lab_run0/single_chip/pointclouds/data').glob('*.bin'))
print('real_files',len(pcfiles))
out['real_frames']=[]
for i in np.linspace(0,len(pcfiles)-1,min(12,len(pcfiles)),dtype=int):
 p=pcfiles[i];pts=np.fromfile(p,np.float32).reshape(-1,5);r=np.hypot(pts[:,0],pts[:,1]);valid=np.isfinite(pts[:,:3]).all(1)&(r>0)&(r<30);pts=pts[valid,:3]
 try:
  x=a.velo_points_2_pano(pts,**cfg);y=b.project(pts)[0];same=x.shape==y.shape
  out['real_frames'].append(dict(path=str(p),original_shape=list(x.shape),current_shape=list(y.shape),different_pixels=int(np.count_nonzero(x!=y)) if same else None,exact_match=bool(np.array_equal(x,y))))
 except Exception as e:out['real_frames'].append(dict(path=str(p),error=str(e)))
target=ROOT/'reports/projection_semantics_20261103';target.mkdir(parents=True,exist_ok=True)
(target/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

