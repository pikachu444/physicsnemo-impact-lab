"""Plot actual deformed geometry from local heldout predictions (no retraining)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

parser=argparse.ArgumentParser()
parser.add_argument("--predictions",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
with np.load(args.predictions) as source:
    coords=source["coords"][0]
    times=source["time"]
    keys=["truth","transolver"]+(["kernel"] if "kernel" in source.files else [])
    fields=[source[key][0].reshape(len(coords),len(times),3) for key in keys]
times=np.concatenate(([0.],times))
fields=[np.concatenate((np.zeros((len(coords),1,3)),field),axis=1) for field in fields]
positions=[coords[:,None,:]+field for field in fields]
vmax=max(np.linalg.norm(field,axis=-1).max() for field in fields)
fig,axes=plt.subplots(1,len(fields),figsize=(4.2*len(fields)+.4,4.2),sharex=True,sharey=True)
all_positions=np.concatenate(positions,axis=0)
scatters=[]
for ax,title in zip(axes,["FEM reference","PhysicsNeMo Transolver","RBF field regression"]):
    ax.scatter(coords[:,1],coords[:,0],s=2,c="lightgray",alpha=.5)
    scatter=ax.scatter(coords[:,1],coords[:,0],c=np.zeros(len(coords)),s=5,cmap="viridis",vmin=0,vmax=vmax)
    scatters.append(scatter)
    ax.set(title=title,xlabel="Y (mm)",ylabel="X (mm)",xlim=(all_positions[:,:,1].min()-35,all_positions[:,:,1].max()+35),ylim=(all_positions[:,:,0].min()-25,all_positions[:,:,0].max()+25))
color_axis=fig.add_axes([.90,.19,.015,.59])
fig.colorbar(scatters[0],cax=color_axis,label="Displacement magnitude (mm)")
heading=fig.suptitle("")
fig.subplots_adjust(left=.06,right=.86,bottom=.13,top=.80,wspace=.23)
def update(frame):
    for scatter,position,field in zip(scatters,positions,fields):
        scatter.set_offsets(position[:,frame,[1,0]])
        scatter.set_array(np.linalg.norm(field[:,frame],axis=-1))
    heading.set_text(f"Heldout Run224 | t={times[frame]:.0f} ms | XY projection, deformation scale 1x\n{len(coords)} evaluated nodes")
    return [*scatters,heading]
update(len(times)-1)
fig.savefig(args.out/"deformed_geometry_comparison.png",dpi=160)
animation=FuncAnimation(fig,update,frames=len(times),interval=250,blit=False)
animation.save(args.out/"deformation_comparison.gif",writer=PillowWriter(fps=4),dpi=110)
plt.close(fig)
print("Saved deformed geometry comparison PNG and GIF")
