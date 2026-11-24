import os,sys,json,numpy as np,pandas as pd
ROOT=sys.argv[1]; REV=sys.argv[2]; sys.path.insert(0,os.path.join(ROOT,'scripts'))
from tailguard_lib import point_stats
te=pd.read_parquet(f'{ROOT}/data/derived/hai2305_test.parquet',columns=['file_id','eligible'])
lab=pd.read_parquet(f'{ROOT}/data/derived/hai2305_test_labels.parquet'); y=lab.label.to_numpy().astype(bool); mask=te.eligible.to_numpy()
t1=mask&(te.file_id.to_numpy()=='hai-test1')
for sc in ('pre','max'):
  F,AP=[],[]
  for s in (17,29,43,71,101):
    z=np.load(f'{REV}/A7_random__W30_inv1.0_phys0.1_phys_regime_bal_s{s}/scores.npz')
    ss=.5*z['p_test']+.5*z['v_test'] if sc=='pre' else np.maximum(z['max_p_test'],z['max_v_test'])
    tt=np.percentile(ss[t1&~y],95); ps=point_stats(ss[t1]>tt,y[t1],ss[t1]); F.append(ps['F1'])
    ps2=point_stats(ss[mask]>tt,y[mask],ss[mask]); AP.append(ps2['AP'])
  print(sc,'F1@5%FPR test1',np.mean(F).round(3),'AP both files',np.mean(AP).round(3))
