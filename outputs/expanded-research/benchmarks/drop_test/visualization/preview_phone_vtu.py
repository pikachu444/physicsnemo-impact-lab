"""Small public PNG preview; large VTUs remain local-only."""
import json
import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
here=Path(__file__).resolve().parent;cache=here.parent/'doe/canonical'
manifest=json.loads((cache/'manifest-full-assembly-corners.json').read_text())
tests=[row for row in manifest['runs'] if row['split']=='test']
reference=np.load(cache/'shared/coordinates_full.npy').astype(np.float64)
targets=[np.load(cache/row['target'],mmap_mode='r')[:,9,:] for row in tests]
norm=Normalize(0,max(float(value[:,3].max()) for value in targets))
fig=plt.figure(figsize=(11.5,6.4),layout='constrained')
axes=[]
for index,(run,target) in enumerate(zip(tests,targets),1):
    ax=fig.add_subplot(1,2,index,projection='3d');axes.append(ax)
    points=reference+target[:,:3]
    ax.scatter(points[:,0],points[:,1],points[:,2],c=target[:,3],norm=norm,cmap='magma',s=.7,depthshade=False,rasterized=True)
    ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]')
    ax.set_zticks([0,5]);ax.tick_params(axis='z',labelsize=8,pad=1)
    ax.zaxis.labelpad=1
    ax.view_init(elev=30,azim=-58);ax.set_box_aspect(np.ptp(reference,axis=0))
    ax.set_title(f"{run['run_id']}\nGPS von Mises max {target[:,3].max():.3f} MPa",fontsize=11)
fig.colorbar(ScalarMappable(norm=norm,cmap='magma'),ax=axes,shrink=.55,label='FEM GPS-derived nodal von Mises [MPa]',pad=.02)
fig.suptitle('Full-phone solid-corner FEM preview: two fixed test conditions\nNominal20 microseconds; geometry = reference + FEM displacement, scale1',fontsize=14)
fig.text(.5,.02,'76065 corner nodes; TETRA10 midsides excluded. Point preview; VTU contains130218 linearized solid cells. No AI prediction arrays.',ha='center',fontsize=9)
fig.savefig(here/'phone-test20us-FEM-3D-preview.png',dpi=180)
plt.close(fig)
