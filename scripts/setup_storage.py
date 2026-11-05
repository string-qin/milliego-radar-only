"""Link an existing data disk into a code checkout without replacing existing data."""
import argparse
import hashlib
import shutil
from pathlib import Path

NAMES=('data','runs','weights','reports','.venv','.cache','.tmp')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage',type=Path,default=Path('/storage/ssd/metaiot_guest/RadarProbMap/baselines/milliego'))
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.project.resolve(strict=True); storage=args.storage.resolve(strict=True)
    # Validate all links before changing any path.
    for name in NAMES:
        path=root/name; target=storage/name
        if not target.is_dir():raise RuntimeError(f'Missing storage directory: {target}; migrate first')
        if path.is_symlink():
            if path.resolve()!=target.resolve():raise RuntimeError(f'Unexpected link: {path}')
        elif path.exists():raise RuntimeError(f'{path} already exists; use migrate_storage.py first')
    for name in NAMES:
        path=root/name
        if not path.is_symlink():path.symlink_to(storage/name,target_is_directory=True)
    # Historical scripts import other scripts below runs/. Versioned originals stay in experiments/.
    for source in (root/'experiments/legacy_runs').rglob('*'):
        if not source.is_file():continue
        target=root/'runs'/source.relative_to(root/'experiments/legacy_runs')
        if target.exists():
            if hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).digest()!=hashlib.sha256(target.read_bytes().replace(b'\r\n',b'\n')).digest():
                raise RuntimeError(f'Existing runtime script differs: {target}; reconcile explicitly')
            continue
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    print('Storage links configured; model weights and cached spectra are not downloaded by this script')

if __name__=='__main__':main()
