import urllib.request,struct as st,numpy as np
from pathlib import Path
u='https://drive.usercontent.google.com/download?id=1ZZl8HQ_Ti0431OCMxv0pH-W1ymF8m7Dm&export=download&confirm=t&sample=cloud32'
N=32*1024**2
with urllib.request.urlopen(urllib.request.Request(u,headers={'Range':f'bytes=0-{N-1}'}),timeout=40) as f:b=f.read(N)
assert b.startswith(b'#ROSBAG')
def fields(h):
 d={};p=0
 while p<len(h):
  n=st.unpack_from('<I',h,p)[0];p+=4;k,v=h[p:p+n].split(b'=',1);d[k]=v;p+=n
 return d
def rec(b,p=0):
 while p+8<len(b):
  n=st.unpack_from('<I',b,p)[0];p+=4
  if p+n+4>len(b):break
  h=fields(b[p:p+n]);p+=n
  m=st.unpack_from('<I',b,p)[0];p+=4
  yield h,b[p:p+m]
  p+=m
conn={};clouds=[]
for h,c in rec(b,13):
 if h.get(b'op')!=bytes([5]) or h.get(b'compression')!=b'none':continue
 for hh,m in rec(c):
  op=hh.get(b'op');cid=hh.get(b'conn')
  if op==bytes([7]):
   if hh.get(b'topic')==b'/mmWaveDataHdl/RScan_middle':
    conn[cid]=fields(m)[b'type'];print('topic type',conn[cid])
  if op!=bytes([2]) or cid not in conn:continue
  p=16+st.unpack_from('<I',m,12)[0]
  if conn[cid]==b'sensor_msgs/PointCloud':
   n=st.unpack_from('<I',m,p)[0];p+=4
   if p+12*n<=len(m):clouds.append(np.frombuffer(m,dtype='<f4',count=3*n,offset=p).reshape(-1,3))
  if conn[cid]==b'sensor_msgs/PointCloud2':
   height,width,nf=st.unpack_from('<III',m,p);p+=12;offs={}
   for _ in range(nf):
    n=st.unpack_from('<I',m,p)[0];p+=4;name=m[p:p+n];p+=n
    off,typ,count=st.unpack_from('<IBI',m,p);p+=9;offs[name]=off
   endian,step,row,n=st.unpack_from('<BIII',m,p);p+=13
   if p+n<=len(m):
    dt=np.dtype('>f4' if endian else '<f4')
    clouds.append(np.column_stack([np.ndarray((height,width),dtype=dt,buffer=m,offset=p+offs[k],strides=(row,step)).ravel() for k in (b'x',b'y',b'z')]))
print('frames',len(clouds))
assert clouds
def summarize(label,clouds):
 rows=[]
 for xyz in clouds:
  x,y,z=xyz[np.isfinite(xyz).all(1)].T
  r=np.hypot(x,y);a=np.degrees(np.arctan2(-y,x));e=np.degrees(np.arctan2(z,r))
  ok=(r>0)&(r<30)&(a>=-60)&(a<60)&(e>=-30)&(e<30)
  pix=np.unique(np.floor((30-e[ok])*64/60)*256+np.floor((a[ok]+60)*256/120)).size
  rows.append([len(r),ok.sum(),np.median(r[ok]) if ok.any() else np.nan,pix])
 print(label,len(rows),np.nanmedian(rows,axis=0))
summarize('author',clouds)
d=np.load('data/coloradar_lab0_native10_v1/labels.npz')
p=Path('/storage/data/metaiot_data/public_dataset/coloradar/12_21_2020_arpg_lab_run0/single_chip/pointclouds/data')
summarize('coloradar',[np.fromfile(p/f'radar_pointcloud_{i}.bin',dtype='<f4').reshape(-1,5)[:,:3] for i in d['frame_ids'][::48]])

