"""Recalibration in the withheld regime from N hours of its normal training data (clean hold-out models),
false alarms counted with the main definition (episodes over the whole sequence, start on a normal row)."""
import os,sys,json,numpy as np,pandas as pd
ROOT,REV,OUT=sys.argv[1:4]; sys.path.insert(0,os.path.join(ROOT,'scripts'))
from tailguard_lib import calibrate_threshold, episodes, event_stats
te=pd.read_parquet(f'{ROOT}/data/derived/hai2305_test.parquet',columns=['segment_id','eligible','regime'])
tr=pd.read_parquet(f'{ROOT}/data/derived/hai2305_train.parquet',columns=['partition','eligible','regime','segment_id'])
lab=pd.read_parquet(f'{ROOT}/data/derived/hai2305_test_labels.parquet'); y=lab.label.to_numpy().astype(bool); ev=lab.event_id.to_numpy()
mask=te.eligible.to_numpy(); reg=te.regime.to_numpy(); seg=te.segment_id.to_numpy()
el=tr.eligible.to_numpy(); rg=tr.regime.to_numpy(); sg=tr.segment_id.to_numpy()
REG=['low','mid','high']; R=[]
for sc in ('pre','max'):
 for kind,tag,inv,li,lp,bal,key in (('TailGuard','clean_tailguard','phys',1.0,0.1,'bal','b0.5'),('GRU','clean_gru','phys',0.0,0.0,'pool','b0.0')):
  for h in REG:
   hi=REG.index(h); nm=mask&~y&(reg==hi); onset=sorted({e for e in pd.unique(ev[mask&y]) if e and reg[np.where(ev==e)[0][0]]==hi}); evh=np.where(np.isin(ev,onset)&mask,ev,'')
   pool=np.where(el&(rg==hi))[0]
   for s in (17,29,43,71,101):
    run=f'{tag}__W30_inv{li}_phys{lp}_{inv}_regime_{bal}_s{s}_hold-{h}'; z=np.load(f'{REV}/{run}/scores.npz'); sm=json.load(open(f'{REV}/{run}/summary.json'))
    pre='max_' if sc=='max' else ''
    if kind=='GRU': st,ss=z[pre+'p_train'],z[pre+'p_test']
    elif sc=='max': st,ss=np.maximum(z['max_p_train'],z['max_v_train']),np.maximum(z['max_p_test'],z['max_v_test'])
    else: st,ss=.5*z['p_train']+.5*z['v_train'],.5*z['p_test']+.5*z['v_test']
    tau=sm['tau'][pre+key]
    for N in (0,1,2,4,8,16,32):
      if N==0: thr=tau
      else:
        rows=pool[:N*3600]
        if len(rows)<N*3600: continue
        thr=np.where(reg==hi,calibrate_threshold(st[rows],sg[rows],len(rows)/3600)[0],tau)
      a=(ss>thr)&mask; E=np.zeros(len(a),bool); E[mask]=episodes(a[mask],seg[mask])
      R.append(dict(score=sc,method=kind,regime=h,seed=s,recal_hours=N,false_episodes=int((E&nm).sum()),FA24=24*(E&nm).sum()/(nm.sum()/3600),
                    detected=event_stats(a[mask],evh[mask])['detected'],attacks=len(onset)))
d=pd.DataFrame(R); d.to_csv(f'{OUT}/recalibration_consistent_runs.csv',index=False)
g=d.groupby(['score','method','regime','recal_hours']).mean(numeric_only=True).drop(columns='seed'); g.to_csv(f'{OUT}/recalibration_consistent.csv')
pd.set_option('display.width',200); pd.set_option('display.max_rows',300); print(g.round(2).to_string())
