"""Public author HDF5: extract existing normalized radar images/labels, not other modalities."""
import sys,json,time,io,urllib.request,hashlib,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_original_h5 import HTTPFile
import numpy as np,h5py
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/original_author_radar_v1'
class FastHTTPFile(HTTPFile):
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n <= self.block and not getattr(self, "_reading_block", False):
            return self._small_read(n)
        if n <= 0:
            return b''
        start=self.pos
        end=start+n-1
        url=f"{self.url}&range_request={start}-{end}"
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'})
        with urllib.request.urlopen(req,timeout=90) as response:
            expected=f"bytes {start}-{end}/{self.size}"
            actual=response.headers.get('Content-Range')
            if response.status != 206 or actual != expected:
                raise IOError(f"Expected {expected}, got HTTP {response.status} with {actual}")
            data=response.read(n+1)
        if len(data)!=n:
            raise IOError(f"Expected {n} bytes, got {len(data)}")
        self.transferred+=n
        self.pos+=n
        return data
    def _small_read(self,n):
        if n<=0:return b''
        chunks=[]
        while n:
            index=self.pos//self.block
            offset=self.pos%self.block
            if index not in self.cache:
                position=self.pos
                self.pos=index*self.block
                self._reading_block=True
                try:
                    self.cache[index]=self.read(min(self.block,self.size-self.pos))
                finally:
                    self.pos=position
                    self._reading_block=False
            block=self.cache[index]
            count=min(n,len(block)-offset)
            chunks.append(block[offset:offset+count])
            self.pos+=count
            n-=count
        return b''.join(chunks)
def process(item):
    split,row=item;stem=Path(row['name']).stem;dest=OUT/split;dest.mkdir(parents=True,exist_ok=True)
    done=dest/(stem+'.json');npz=dest/(stem+'.npz')
    if done.exists() and npz.exists():return json.loads(done.read_text())
    size=row['metadata_excerpt'][13]
    url='https://drive.usercontent.google.com/download?id='+row['id']+'&export=download&confirm=t'
    for attempt in range(3):
        try:
            source=FastHTTPFile(url,size)
            with h5py.File(source,'r') as f:
                images=f['mmwave_middle_data'][:];labels=f['label_data'][0];timestamps=f['timestamp'][0]
            assert images.shape[0]==len(labels)+1==len(timestamps)
            assert images.shape[1:]==(1,64,256,1)
            assert np.isfinite(images).all() and np.isfinite(labels).all()
            assert np.all(np.diff(timestamps)>0)
            np.savez_compressed(npz,images=images,labels=labels,timestamps=timestamps)
            rec={'split':split,'sequence':stem,'file_id':row['id'],'source_size':size,'download_bytes':source.transferred,'frames':len(images),'pairs':len(labels),'mean_dt_s':float(np.diff(timestamps).mean()/1e9),'image_min':float(images.min()),'image_max':float(images.max()),'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
            done.write_text(json.dumps(rec,indent=2));print('EXTRACTED',json.dumps(rec),flush=True);return rec
        except Exception as e:
            print('RETRY',split,stem,attempt,str(e),flush=True)
            if attempt==2:raise
def main():
    OUT.mkdir(parents=True,exist_ok=True);items=[]
    for split in ['train','test']:
        rows=json.loads((ROOT/('reports/original_source_check_20261103/original_'+split+'_listing.json')).read_text())['entries']
        items.extend((split,r) for r in rows if r['name'].endswith('.h5'))
    recs=[]; failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(process,item):item for item in items}
        for job in concurrent.futures.as_completed(pending):
            split,row=pending[job]
            try:
                recs.append(job.result())
            except Exception as error:
                failures.append({'split':split,'name':row['name'],'error':str(error)})
    (OUT/'availability.json').write_text(json.dumps({'available':len(recs),'missing':failures},indent=2))
    if failures:
        raise SystemExit(f"Extracted {len(recs)} sequences; {len(failures)} unavailable. See availability.json.")
    assert len(recs)==57 and len([x for x in recs if x['split']=='train'])==49
    (OUT/'manifest.json').write_text(json.dumps({'source':'Public author milliEgo_original_dataset Google Drive','normalization':'Already normalized in author HDF5. Preserve as provided.','records':recs,'all_requested_sequences_complete':True},indent=2))
    print('COMPLETE',len(recs),'pairs',sum(x['pairs'] for x in recs),'bytes',sum(x['download_bytes'] for x in recs),flush=True)
if __name__=='__main__':main()
