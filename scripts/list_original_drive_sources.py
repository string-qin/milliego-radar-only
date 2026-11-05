import json,re,codecs,urllib.request,concurrent.futures
from pathlib import Path
root=Path("/home/metaiot_guest/Workspace/qinzx/milliEgo/reports/original_source_check_20261103")
def decode(s):
    m=re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'(.*?)';",s,re.S)
    if not m:raise ValueError("Public folder payload not found")
    a=json.loads(codecs.decode(m[1],"unicode_escape"))
    return a[0]
def fetch(item):
    name,fid=item;rec={"folder_name":name,"folder_id":fid}
    try:
        url="https://drive.google.com/drive/folders/"+fid
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"milliEgo-reproduction-read-only"}),timeout=25) as f:s=f.read(4000000).decode()
        (root/(name+".html")).write_text(s);rows=decode(s);rec["entries"]=[{"id":x[0],"name":x[2],"mime":x[3],"metadata_numeric":[[i,v] for i,v in enumerate(x[:30]) if isinstance(v,(float,int)) and v>0],"metadata_excerpt":x[:16]} for x in rows]
    except Exception as e:rec["error"]=repr(e)
    (root/(name+"_listing.json")).write_text(json.dumps(rec,indent=2));return rec
rootrows=decode((root/"drive_folder.html").read_text());print("ROOT",json.dumps([{"id":x[0],"name":x[2],"mime":x[3]} for x in rootrows],ensure_ascii=False))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,[("original_dataset","1fScQD8O9oxLEUfuHILWmALmemi0a6a78"),("models","112XjxwUDN_HpRCTaAPstt3k8ZpJgmyDE")]))
for r in records:print(json.dumps(r,ensure_ascii=False),flush=True)

