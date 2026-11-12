# Kaggle notebook cell used for Step 5 (training) on 2 x Tesla T4.
# Attach the repository (with data/derived from Step 2 and runs/baselines from Step 3) as a Kaggle dataset.
# A Kaggle session stops after 12 h: --max-hours 11 stops starting new runs; re-run the cell in a new session with the
# previous notebook output attached as an input and finished runs are skipped (each run checkpoints every epoch).
# The original cell also patched checkpoint RNG loading for GPU; that fix is now in scripts/tailguard_train.py.
import os, glob, shutil
hits = glob.glob('/kaggle/input/**/scripts/tailguard_train.py', recursive=True)
src = os.path.dirname(os.path.dirname(hits[0])); dst = '/kaggle/working/tailguard'
os.makedirs(dst, exist_ok=True)
for d in ['data', 'manifests']:
    if not os.path.exists(f'{dst}/{d}'): os.symlink(f'{src}/{d}', f'{dst}/{d}')
if os.path.islink(f'{dst}/scripts'): os.unlink(f'{dst}/scripts')
if not os.path.exists(f'{dst}/scripts'): shutil.copytree(f'{src}/scripts', f'{dst}/scripts')
# newest saved runs: the runs/train folder with the most finished runs among all inputs
cands = [p for p in glob.glob('/kaggle/input/**/runs/train', recursive=True)]
best = max(cands, key=lambda p: len(glob.glob(f'{p}/*/summary.json'))) if cands else None   # None on the first session
print('runs/train candidates:', {p: len(glob.glob(f'{p}/*/summary.json')) for p in cands}); print('using', best)
if not os.path.exists(f'{dst}/runs'):
    if best: shutil.copytree(os.path.dirname(best), f'{dst}/runs')
    else: os.makedirs(f'{dst}/runs/train')
    shutil.copytree(f'{src}/runs/deep', f'{dst}/runs/deep', dirs_exist_ok=True) if os.path.exists(f'{src}/runs/deep') else None
    shutil.copytree(f'{src}/runs/baselines', f'{dst}/runs/baselines', dirs_exist_ok=True)
os.chdir(dst)
print(len(glob.glob('runs/train/*/summary.json')), 'finished runs available')
import subprocess
subprocess.run(["nvidia-smi", "-L"])
subprocess.run(["python", "scripts/tailguard_train.py", "--stage", "all", "--max-hours", "11"], check=True)