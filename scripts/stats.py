import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import save_json
p=argparse.ArgumentParser();p.add_argument('--trace-dir',default='traces');p.add_argument('--output',default='stats');a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
files=[x for x in sorted(Path(a.trace_dir).glob('*.npz')) if 'repeat' not in x.name]
assert files,'No real trace files exist; refusing to fabricate plots'
summary=[]
for path in files:
 with np.load(path,allow_pickle=False) as z:
  ids=z['expert_ids'];loads=np.stack([np.bincount(ids[:,l,:].ravel(),minlength=128) for l in range(48)])
 mean=loads.mean(1);maximum=loads.max(1);jain=loads.sum(1)**2/(128*(loads.astype(float)**2).sum(1))
 with (out/(path.stem+'.csv')).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['layer_id','expert_id','selection_count','max_load','mean_load','max_over_mean','jain_fairness'])
  for l in range(48):
   for e in range(128):w.writerow([l,e,loads[l,e],maximum[l],mean[l],maximum[l]/mean[l],jain[l]])
 fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(loads/mean[:,None],aspect='auto',vmin=0);fig.colorbar(im,ax=ax,label='Selections / layer mean');ax.set(xlabel='Expert ID',ylabel='MoE layer',title=f'{path.stem}: {ids.shape[0]} valid tokens');fig.tight_layout();fig.savefig(out/(path.stem+'.png'),dpi=150);plt.close(fig)
 summary.append({'batch_id':path.stem,'tokens':ids.shape[0],'mean_layer_jain':float(jain.mean()),'max_layer_load_ratio':float((maximum/mean).max())})
save_json(out/'summary.json',summary)
fig,ax=plt.subplots(figsize=(9,4))
for prefix in ['single','batch','minimal']:
 rows=sorted([s for s in summary if s['batch_id'].startswith(prefix)],key=lambda s:s['tokens'])
 if rows:ax.plot([s['tokens'] for s in rows],[s['mean_layer_jain'] for s in rows],'o-',label=prefix)
ax.set(xscale='log',xlabel='Valid prefill tokens',ylabel='Mean layer Jain fairness');ax.legend();fig.tight_layout();fig.savefig(out/'fairness_by_scale.png',dpi=150)
