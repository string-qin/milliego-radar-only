from pathlib import Path
import h5py,numpy as np,json,hashlib
out=Path('data/original_author_radar_v1/test')
out.mkdir(exist_ok=True)
with h5py.File('data/turtle_seq_52.h5') as f:
    images=f['mmwave_middle_data'][:]
    labels=f['label_data'][0]
    timestamps=f['timestamp'][0]
assert images.shape==(len(labels)+1,1,64,256,1)
assert len(timestamps)==len(images) and np.all(np.diff(timestamps)>0)
assert np.isfinite(images).all() and np.isfinite(labels).all()
p=out/'turtle_seq_52.npz'
np.savez_compressed(p,images=images,labels=labels,timestamps=timestamps)
(out/'turtle_seq_52.json').write_text(json.dumps(dict(sequence=p.stem,npz_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
