# TailGuard refitted hold-out, A7 and HAI 22.04 runs on Kaggle (2 x Tesla T4). Paste into one cell; "Save Version" -> "Save & Run All (Commit)".
# Inputs to attach: (1) your existing dataset <your-kaggle-account>/tailguard-data (HAI 23.05 derived data), (2) the new dataset made from
# the folder tailguard_holdout_a7 (the code bundle), and - from the second session on - (3) the output of the previous version of THIS notebook.
# Finished runs are skipped; every run checkpoints each epoch.
# Work (in priority order, split over both GPUs):
#   1. HAI 23.05 clean regime hold-out: TailGuard, matched GRU, A2, A10 x 3 regimes x 5 seeds; A7 random constraints x 5
#   2. HAI 22.04 confirmation: TailGuard, matched GRU, A1 x 5 seeds (frozen protocol)
#   3. HAI 22.04 deep baselines: USAD, TranAD (GPU 0) and GDN (GPU 1), 5 seeds each
#   4. HAI 22.04 classical baselines (CPU, after the GPU-0 work; never alongside training - memory)
import os, glob, shutil, subprocess, time, sys
T0 = time.time(); DEADLINE = int(T0 + 11.2 * 3600)          # Kaggle stops a session at 12 h; leave time to save outputs
import pandas as pd
def find(rel, maxdepth=6):
    """Paths /kaggle/input/<up to maxdepth folders>/<rel> (fast: fixed-depth patterns instead of a full recursive walk)."""
    return [p for d in range(maxdepth + 1) for p in glob.glob('/kaggle/input/' + '*/' * d + rel)]
src = [os.path.dirname(p) for p in find('holdout_a7_cell.py') if os.path.exists(os.path.join(os.path.dirname(p), 'scripts', 'holdout_a7_runs.py'))][0]
d2305 = os.path.dirname(find('data/derived/hai2305_train.parquet')[0])          # your existing tailguard dataset
print('code bundle:', src); print('HAI 23.05 data:', d2305)
W = '/kaggle/working'
for ds in ('hai2305', 'hai2204'):
    dst = f'{W}/{ds}'; os.makedirs(f'{dst}/data/derived', exist_ok=True)
    if ds == 'hai2305':
        for f in ('hai2305_train.parquet', 'hai2305_test.parquet'):
            if not os.path.exists(f'{dst}/data/derived/{f}'): os.symlink(f'{d2305}/{f}', f'{dst}/data/derived/{f}')
    else:
        for name in ('hai2204_train', 'hai2204_test'):
            if not os.path.exists(f'{dst}/data/derived/{name}.parquet'):
                parts = sorted(glob.glob(f'{src}/hai2204/data/derived/{name}.part*.parquet'))
                pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(f'{dst}/data/derived/{name}.parquet', index=False)
    if not os.path.exists(f'{dst}/manifests'): shutil.copytree(f'{src}/{ds}/manifests', f'{dst}/manifests')
    if not os.path.exists(f'{dst}/scripts'): shutil.copytree(f'{src}/scripts', f'{dst}/scripts')
    os.makedirs(f'{dst}/runs', exist_ok=True)
    shutil.copytree(f'{src}/{ds}/runs', f'{dst}/runs', dirs_exist_ok=True)
    # resume: copy finished/partial runs from earlier versions of this notebook (newest = most summaries)
    for sub in ('holdout_a7', 'train', 'deep_rescore', 'baselines'):
        cands = [p for p in find(f'{ds}/runs/{sub}') if not p.startswith(src)]
        if cands:
            best = max(cands, key=lambda p: len(glob.glob(f'{p}/**/summary.json', recursive=True)) + len(glob.glob(f'{p}/**/ckpt.pt', recursive=True)))
            shutil.copytree(best, f'{dst}/runs/{sub}', dirs_exist_ok=True); print('resumed', ds, sub, 'from', best)
subprocess.run(['nvidia-smi', '-L'])

def job(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    log = open(f'{W}/gpu{gpu}.log', 'a')
    steps = [(f'{W}/hai2305', {}, ['python', 'scripts/holdout_a7_runs.py', '--part', str(gpu), '--parts', '2']),
             (f'{W}/hai2204', {'TAILGUARD_PREFIX': 'hai2204'}, ['python', 'scripts/confirm_2204.py', '--part', str(gpu), '--parts', '2'])]
    deep = ['USAD', 'TranAD'] if gpu == 0 else ['GDN']
    steps += [(f'{W}/hai2204', {'TAILGUARD_PREFIX': 'hai2204'}, ['python', 'scripts/step7_deep_rescore.py', '--model', m]) for m in deep]
    if gpu == 0:   # classical baselines last, alone on this chain (they need ~8 GB RAM; running them alongside training ran out of memory)
        steps.append((f'{W}/hai2204', {'TAILGUARD_PREFIX': 'hai2204', 'CUDA_VISIBLE_DEVICES': ''}, ['python', 'scripts/step3_baselines.py']))
    script = ' ; '.join(f"cd {d} && {' '.join(f'{k}={v}' for k, v in e.items())} timeout $(( {DEADLINE} - $(date +%s) )) {' '.join(c)}" for d, e, c in steps)
    return subprocess.Popen(['bash', '-c', script], env=env, stdout=log, stderr=subprocess.STDOUT)
procs = [job(0), job(1)]
while any(p.poll() is None for p in procs):
    time.sleep(600)
    mem = subprocess.run(['free', '-g'], capture_output=True, text=True).stdout.splitlines()[1].split()
    done = len(glob.glob(f'{W}/hai2305/runs/holdout_a7/*/summary.json')) + len(glob.glob(f'{W}/hai2204/runs/train/*/summary.json'))
    for g in (0, 1):
        tail = open(f'{W}/gpu{g}.log').read().splitlines()[-1:] if os.path.exists(f'{W}/gpu{g}.log') else []
        print(f'[{(time.time()-T0)/3600:.1f} h | RAM used {mem[2]}/{mem[1]} GB | {done} runs done] GPU{g}:', tail, flush=True)
    if time.time() > DEADLINE + 300: break
n = lambda p: len(glob.glob(p, recursive=True))
print('clean hold-out + A7 finished:', n(f'{W}/hai2305/runs/holdout_a7/*/summary.json'), 'of 65')
print('HAI 22.04 GRU runs finished:', n(f'{W}/hai2204/runs/train/*/summary.json'), 'of 15')
print('HAI 22.04 classical baselines:', n(f'{W}/hai2204/runs/baselines/*/summary.json'), 'of 4')
print('HAI 22.04 deep runs finished:', n(f'{W}/hai2204/runs/deep_rescore/*/*/summary.json'), 'of 15')
print('If any count is short, run a new version with this notebook\'s output attached; finished runs are skipped.')
