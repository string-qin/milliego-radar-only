import json,os,subprocess,concurrent.futures,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=Path("/storage/data/metaiot_data/public_dataset")
dest=ROOT/"reports/dataset_parts";dest.mkdir(exist_ok=True)
jobs=[]
# Names only, no recursive traversal or per-entry stat.
for n in sorted(os.listdir(SRC/"coloradar")):
    if "_run" in n:jobs.append(("ColoRadar",n,"1" if n in ["12_21_2020_arpg_lab_run0","2_24_2021_aspen_run0","2_28_2021_outdoors_run0"] else "0",18))
for n in sorted(os.listdir(SRC/"KRadar")):
    if n.isdigit():jobs.append(("K-Radar",n,"1" if n=="1" else "0",45 if n=="1" else 18))
for n in sorted(os.listdir(SRC/"RaDelft")):
    if n.startswith("Scene"):jobs.append(("RaDelft",n,"1",35))
def check(job):
    kind,seq,payload,limit=job;key=kind+"_"+seq;target=dest/(key+".json")
    env=dict(os.environ,AUDIT_KIND=kind,AUDIT_SEQUENCE=seq,AUDIT_PAYLOAD=payload,AUDIT_OUTPUT=str(target))
    start=time.monotonic()
    with (dest/(key+".log")).open("w") as log:
        p=subprocess.Popen([str(ROOT/".venv/bin/python"),"-u",str(ROOT/"scripts/audit_dataset_worker.py")],env=env,stdout=log,stderr=log,start_new_session=True)
        try:rc=p.wait(timeout=limit);status="passed" if rc==0 else "error"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:pass
            rc=None;status="timeout"
    return {"dataset":kind,"sequence":seq,"payload_sampled":payload=="1","status":status,"returncode":rc,"seconds":time.monotonic()-start,"result":str(target) if target.exists() else None,"log":str(dest/(key+".log"))}
index=ROOT/"reports/dataset_audit_index.json"
old=json.loads(index.read_text()) if index.exists() else {"records":[]}
if index.exists() and not (ROOT/"reports/dataset_audit_initial_index.json").exists():(ROOT/"reports/dataset_audit_initial_index.json").write_text(index.read_text())
records=[r for r in old["records"] if r["status"]=="passed"]
passed={(r["dataset"],r["sequence"]) for r in records}
pending_jobs=[j for j in jobs if (j[0],j[1]) not in passed]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(check,j) for j in pending_jobs]
    for f in concurrent.futures.as_completed(futures):
        rec=f.result();records.append(rec)
        summary={"scope":"Current all-sequence metadata attempts; sampled radar payloads. Timeouts/errors are NOT passes.","attempted_total":len(jobs),"completed":len(records),"records":sorted(records,key=lambda r:(r["dataset"],r["sequence"]))}
        (ROOT/"reports/dataset_audit_index.json").write_text(json.dumps(summary,indent=2))
        print(rec["dataset"],rec["sequence"],rec["status"],round(rec["seconds"],2),flush=True)
print("COMPLETE",flush=True)

