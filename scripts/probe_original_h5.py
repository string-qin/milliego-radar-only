import io,urllib.request,json,sys
from pathlib import Path
import h5py,numpy as np
class HTTPFile(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={};self.transferred=0;self.block=1024*1024
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def readinto(self,b):
  v=self.read(len(b));b[:len(v)]=v;return len(v)
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos);parts=[]
  while n>0:
   i=self.pos//self.block
   if i not in self.cache:
    start=i*self.block;end=min(start+self.block,self.size)-1
    req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=45) as r:
     assert r.status==206,'Server ignored byte range'
     x=r.read();assert len(x)==end-start+1
    self.cache[i]=x;self.transferred+=len(x)
   x=self.cache[i];a=self.pos%self.block;k=min(n,len(x)-a)
   assert k>0
   parts.append(x[a:a+k]);self.pos+=k;n-=k
  return b''.join(parts)
def main():
 url='https://drive.usercontent.google.com/download?id=1xenb4uBQoIRA8XRjmBnQumkM0VP1IXe4&export=download&confirm=t'
 rf=HTTPFile(url,489607624);out=Path('data/original_author_probe');out.mkdir(parents=True,exist_ok=True)
 with h5py.File(rf,'r') as f:
  info={k:{'shape':list(v.shape),'dtype':str(v.dtype),'compression':v.compression} for k,v in f.items() if isinstance(v,h5py.Dataset)}
  print(json.dumps(info),flush=True)
  selected={}
  for k in f.keys():
   if any(s in k.lower() for s in ['mmwave_middle','label','timestamp']):
    selected[k]=f[k][:]
    print('extracted',k,selected[k].shape,flush=True)
  np.savez_compressed(out/'turtle_seq_36_radar_labels.npz',**selected)
 report={'source_file':'turtle_seq_36.h5','source_file_size':rf.size,'bytes_downloaded':rf.transferred,'datasets':info,'selected':{k:{'shape':list(v.shape),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),'finite':bool(np.isfinite(v).all())} for k,v in selected.items()}}
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

