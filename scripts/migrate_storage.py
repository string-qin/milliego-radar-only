"""Copy, checksum-verify, then replace project artifact directories with symlinks."""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path

DEFAULT_PROJECT='/home/metaiot_guest/Workspace/qinzx/milliEgo'
DEFAULT_STORAGE='/storage/ssd/metaiot_guest/RadarProbMap/baselines/milliego'
NAMES=('data','runs','weights','reports','.venv','.cache','.tmp')

def run(command):
    subprocess.run(command,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(DEFAULT_PROJECT))
    parser.add_argument('--storage',type=Path,default=Path(DEFAULT_STORAGE))
    parser.add_argument('--apply',action='store_true',help='Copy and cut over; default only checks paths')
    args=parser.parse_args()
    project=args.project.resolve(strict=True)
    storage=args.storage.resolve(strict=True)
    if project==storage or project in storage.parents or storage in project.parents:
        raise RuntimeError('Source and storage must be separate directory trees')
    if not os.access(storage,os.W_OK|os.X_OK):
        raise PermissionError(f'{storage} is not writable; source is unchanged')
    for name in NAMES:
        source=project/name; dest=storage/name
        if source.is_symlink():
            if source.resolve()!=dest.resolve():raise RuntimeError(f'Unexpected existing link: {source}')
            continue
        if not source.is_dir():raise RuntimeError(f'Missing source directory: {source}')
        print('PLAN',source,'->',dest,flush=True)
    if not args.apply:return
    log=storage/'migration_20261105.json'
    records=[]
    if log.exists():records=json.loads(log.read_text())
    for name in NAMES:
        source=project/name; dest=storage/name
        if source.is_symlink():continue
        dest.mkdir(exist_ok=True)
        backup=project/(name+'.migration_backup')
        if backup.exists():raise RuntimeError(f'Existing migration backup: {backup}')
        print('COPY',name,flush=True)
        run(['rsync','-aH','--',str(source)+'/',str(dest)+'/'])
        # Read checksums from both copies. Any missing/different/extra entry blocks deletion.
        verify=subprocess.run(['rsync','-aHnci','--delete','--',str(source)+'/',str(dest)+'/'],check=True,capture_output=True,text=True)
        if verify.stdout.strip():raise RuntimeError('Checksum verification differs: '+verify.stdout[:2000])
        files=sum(1 for p in source.rglob('*') if p.is_file() and not p.is_symlink())
        source.rename(backup)
        try:
            source.symlink_to(dest,target_is_directory=True)
            assert source.resolve()==dest.resolve()
        except BaseException:
            if source.is_symlink():source.unlink()
            backup.rename(source)
            raise
        # Only remove the verified original directory under the explicitly selected project.
        assert backup.parent.resolve()==project and backup.name==name+'.migration_backup'
        assert not backup.is_symlink() and backup.resolve().parent==project
        shutil.rmtree(backup)
        records.append(dict(source=str(source),target=str(dest),files=files,checksum_verified=True,time=time.time()))
        temporary=log.with_suffix('.json.tmp'); temporary.write_text(json.dumps(records,indent=2)); temporary.replace(log)
        print('MIGRATED',name,files,'files',flush=True)
    # Preserve completed experiments that were temporarily stored in RAM after quota was reached.
    for name in ('milliego-author-spectrum-20261105','milliego-author-scratch-20261105','milliego-turn-c00ez_gl','milliego-angle-comparison-20261105','milliego-batch.Cr6G7f'):
        source=Path('/dev/shm')/name
        if not source.is_dir():continue
        dest=storage/'ram_experiments'/name; dest.mkdir(parents=True,exist_ok=True)
        run(['rsync','-aH','--',str(source)+'/',str(dest)+'/'])
        result=subprocess.check_output(['rsync','-aHnci','--delete','--',str(source)+'/',str(dest)+'/'],text=True)
        if result.strip():raise RuntimeError(f'RAM experiment copy differs: {name}')
    print('COMPLETE; RAM originals retained; public raw datasets untouched',flush=True)

if __name__=='__main__':main()
