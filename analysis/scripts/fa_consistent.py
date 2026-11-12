"""Per-regime false alarms with the paper's main definition (episodes over the whole test sequence; false if the
episode starts on a normal row; attributed to the regime of its start row), for in-distribution and hold-out models."""
import os,sys,json,numpy as np,pandas as pd
ROOT,REV,OUT=sys.argv[1:4]; sys.path.insert(0,os.path.join(ROOT,'scripts'))
from tailguard_lib import calibrate_threshold, episodes, event_stats
te=pd.read_parquet(f'{ROOT}/data/derived/hai2305_test.parquet',columns=['file_id','segment_id','eligible','regime'])
tr=pd.read_parquet(f'{ROOT}/data/derived/hai2305_train.parquet',columns=['partition','eligible','regime','segment_id'])
lab=pd.read_parquet(f'{ROOT}/data/derived/hai2305_test_labels.parquet'); y=lab.label.to_numpy().astype(bool); ev=lab.event_id.to_numpy()
mask=te.eligible.to_numpy(); reg=te.regime.to_numpy(); seg=te.segment_id.to_numpy()
p=tr.partition.to_numpy(); el=tr.eligible.to_numpy(); rg=tr.regime.to_numpy(); sg=tr.segment_id.to_numpy(); calm=el&(p=='cal')
REG=['low','mid','high']; SEEDS=[17,29,43,71,101]
st_ep=np.zeros(len(mask),bool)
def per_regime(ss,tau):
    a=(ss>tau)&mask; st=np.zeros(len(a),bool); st[mask]=episodes(a[mask],seg[mask]); out={}
    for r in range(3):
        nm=mask&~y&(reg==r); out[REG[r]]=(int((st&nm).sum()), nm.sum()/3600)
        onset=sorted({e for e in pd.unique(ev[mask&y]) if e and reg[np.where(ev==e)[0][0]]==r}); evh=np.where(np.isin(ev,onset)&mask,ev,'')
        out[REG[r]]+= (event_stats(a[mask],evh[mask])['detected'],len(onset))
    return out
def rid(t,s,li=1.0,lp=0.1,bal='bal',inv='phys',h=''): return f'{t}__W30_inv{li}_phys{lp}_{inv}_regime_{bal}_s{s}{h}'
F={'pre':{'tailguard':lambda z:(.5*z['p_train']+.5*z['v_train'],.5*z['p_test']+.5*z['v_test']),'gru':lambda z:(z['p_train'],z['p_test'])},
   'max':{'tailguard':lambda z:(np.maximum(z['p_train'],z['v_train']),np.maximum(z['p_test'],z['v_test'])),'gru':lambda z:(z['p_train'],z['p_test'])}}
FR={'pre':{'tailguard':lambda z:(.5*z['p_train']+.5*z['v_train'],.5*z['p_test']+.5*z['v_test']),'gru':lambda z:(z['p_train'],z['p_test'])},
    'max':{'tailguard':lambda z:(np.maximum(z['max_p_train'],z['max_v_train']),np.maximum(z['max_p_test'],z['max_v_test'])),'gru':lambda z:(z['max_p_train'],z['max_p_test'])}}
R=[]
for sc,tdir in (('pre','runs/train'),('max','runs_max/train')):
  for kind in ('tailguard','gru'):
    for s in SEEDS:
      # in-distribution model (all regimes in training)
      run=rid('search' if s==17 else 'tailguard',s) if kind=='tailguard' else rid('gru_erm',s,0.0,0.0,'pool')
      z=np.load(f'{ROOT}/{tdir}/{run}/scores.npz'); st,ss=F[sc][kind](z); tau=calibrate_threshold(st[calm],sg[calm],calm.sum()/3600)[0]
      for r,(e,h,d,n) in per_regime(ss,tau).items(): R.append(dict(score=sc,model=kind,variant='all regimes',seed=s,regime=r,ep=e,h=h,det=d,n=n))
      for h_ in REG:
        hi=REG.index(h_)
        # original (partial) hold-out
        run=rid('hold_tailguard',s,h=f'_hold-{h_}') if kind=='tailguard' else rid('hold_gru',s,0.0,0.0,'pool',h=f'_hold-{h_}')
        z=np.load(f'{ROOT}/{tdir}/{run}/scores.npz'); st,ss=F[sc][kind](z); cm=calm&(rg!=hi)
        tau=calibrate_threshold(st[cm],sg[cm],cm.sum()/3600)[0]; e,hh,d,n=per_regime(ss,tau)[h_]
        R.append(dict(score=sc,model=kind,variant='partial hold-out',seed=s,regime=h_,ep=e,h=hh,det=d,n=n))
        # refitted hold-out runs
        run=rid('clean_tailguard',s,h=f'_hold-{h_}') if kind=='tailguard' else rid('clean_gru',s,0.0,0.0,'pool',h=f'_hold-{h_}')
        sm=json.load(open(f'{REV}/{run}/summary.json')); z=np.load(f'{REV}/{run}/scores.npz'); st,ss=FR[sc][kind](z)
        key=('max_' if sc=='max' else '')+('b0.5' if kind=='tailguard' else 'b0.0'); e,hh,d,n=per_regime(ss,sm['tau'][key])[h_]
        R.append(dict(score=sc,model=kind,variant='clean hold-out',seed=s,regime=h_,ep=e,h=hh,det=d,n=n))
d=pd.DataFrame(R); d.to_csv(f'{OUT}/regime_fa_consistent_runs.csv',index=False)
g=d.groupby(['score','model','regime','variant']).agg(ep=('ep','mean'),h=('h','first'),det=('det','mean'),n=('n','first')).reset_index()
g['FA24']=24*g.ep/g.h; g['recall']=g.det/g.n; g.to_csv(f'{OUT}/regime_fa_consistent.csv',index=False)
pd.set_option('display.width',200); print(g.round(2).to_string())
