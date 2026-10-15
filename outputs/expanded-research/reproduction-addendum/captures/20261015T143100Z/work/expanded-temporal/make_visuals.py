"""Scientific figures from saved held-out FEM and actual model predictions."""
from pathlib import Path
import json, os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
records={}
for name in ['time_conditional','ar_rollout_training','one_step_teacher_forcing']:
    records[name]=json.loads((OUT/'coupon13'/name/'result.json').read_text())
colors=['#165c98','#b8641b','#4d8954']
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for (name,r),color in zip(records.items(),colors):
    hist=json.loads((OUT/'coupon13'/name/'history.json').read_text())
    axs[0].plot([x['epoch'] for x in hist],[x['validation_rollout_mse'] for x in hist],label=name,color=color)
    npz=np.load(OWN/'checkpoints'/name/'test-predictions.npz')
    err=npz['prediction'][...,:3].astype('float64')-npz['truth'][...,:3]
    rmse=np.sqrt((err**2).mean((0,1,3)))
    axs[1].plot(range(1,11),rmse,label=name,color=color,marker='o',ms=3)
axs[0].set(xlabel='Epoch',ylabel='Validation rollout normalized MSE',yscale='log',title='Different C7 / C3 tasks; not a controlled ranking')
axs[1].set(xlabel='Nominal time (ms)',ylabel='Displacement component RMSE (mm)',title='Held-out FEM conditions, free rollout')
for ax in axs: ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'coupon13-temporal-learning-and-errors.png',dpi=180)
plt.close(fig)

data=np.load(OWN/'checkpoints/time_conditional/test-predictions.npz')
pred,truth=data['prediction'],data['truth']
xy=np.load(ROOT/'work/expanded-structural-coupon/cache-initial13-nodal7/coordinates.npy')[:,:2]
# First held-out condition, fixed in advance by the frozen split order.
fig,axs=plt.subplots(3,3,figsize=(13.5,5),layout='constrained')
for row,(field,title,unit) in enumerate([(3,'Von Mises stress','MPa'),(4,'Equivalent plastic strain','1'),(5,'Total strain principal major','1')]):
    actual=truth[0,:,-1,field]; estimate=pred[0,:,-1,field]
    low=float(min(actual.min(),estimate.min())); high=float(max(actual.max(),estimate.max()))
    for col,(value,label) in enumerate([(actual,'FEM'),(estimate,'AI'),(estimate-actual,'AI minus FEM')]):
        ax=axs[row,col]
        if col==2:
            extent=float(np.abs(value).max()) or 1.
            im=ax.scatter(xy[:,0],xy[:,1],c=value,s=5,cmap='RdBu_r',vmin=-extent,vmax=extent)
        else: im=ax.scatter(xy[:,0],xy[:,1],c=value,s=5,cmap='viridis',vmin=low,vmax=high)
        ax.set_title(f'{title}\n{label}' if col==0 else label,fontsize=9)
        ax.set_xlabel('X (mm)',fontsize=8);ax.set_ylabel('Y (mm)',fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal')
        cb=fig.colorbar(im,ax=ax,shrink=.8,label=unit)
        cb.ax.tick_params(labelsize=8)
fig.suptitle('Time-conditional C7, test condition1 at 10ms; area-projected nodal FEM invariants',fontsize=11)
fig.savefig(OUT/'coupon13-timeconditional-stress-strain-fields.png',dpi=180)
plt.close(fig)
print(json.dumps({'figures':['coupon13-temporal-learning-and-errors.png','coupon13-timeconditional-stress-strain-fields.png']}))
