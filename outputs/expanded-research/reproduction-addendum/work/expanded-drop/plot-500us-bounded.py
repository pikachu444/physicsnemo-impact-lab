"""Render compact FEM figures in a separate CPU process with a 200 MiB guard."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
from resource_gate import guard as resource_guard
def guard(stage):
    result=resource_guard(stage,minimum_available=768*1024**2)
    result['configured_minimum_available_bytes']=768*1024**2
    return result
guard('before_plot_imports')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/expanded-research/benchmarks/drop_test'
parser=argparse.ArgumentParser()
parser.add_argument('--destination',type=Path,default=BASE/'500us-final')
args=parser.parse_args();destination=args.destination
stats=json.loads((destination/'frame-field-statistics.json').read_text())
summary=json.loads((destination/'summary.json').read_text())
time_us=[r['actual_time_s']*1e6 for r in stats]
fig,axes=plt.subplots(1,3,figsize=(12.5,3.7),layout='constrained')
for ax,key,label in zip(axes,['native_cell_vm_max_MPa','total_strain_all_valid_IP_max_frobenius','native_cell_epsp_max'],['Native cell von Mises maximum [MPa]','All valid IP total strain max. Frobenius [1]','Native cell EPSP maximum [1]']):
    ax.plot(time_us,[r[key] for r in stats],color='#1a607e',marker='o',markersize=2.5)
    ax.axvline(20,color='grey',ls='--',lw=.8);ax.axvline(100,color='grey',ls=':',lw=.8)
    ax.grid(alpha=.2);ax.set(xlabel='Actual saved frame time [microseconds]',ylabel=label)
fig.suptitle('Single-center FEM diagnostic: native solid fields; AI labels remain 2-20 microseconds',fontsize=11)
fig.savefig(destination/'phone-500us-field-trends.png',dpi=160);plt.close(fig);gc.collect()
guard('field_trends_plot_complete')
with (destination/'global-history-0-to500us.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
times=np.array([float(r['time'])*1e6 for r in rows])
normal=np.asarray(json.loads((BASE/'contact-time-estimate.json').read_text())['plane_unit_normal'])
velocity=np.array([[float(r[k]) for k in ['X-MOMENTUM','Y-MOMENTUM','Z-MOMENTUM']] for r in rows])@normal/np.array([float(r['MASS'])*1000 for r in rows])
accounted=np.array([sum(float(r[k]) for k in ['KINETIC ENERGY','INTERNAL ENERGY','ROTATION ENERGY','SPRING ENERGY','CONTACT ENERGY','HOURGLASS ENERGY'])-float(r['EXTERNAL WORK']) for r in rows])*.001
fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
for key in ['KINETIC ENERGY','INTERNAL ENERGY','PLASTIC WORK']:
    axes[0].plot(times,[float(r[key])*.001 for r in rows],label=key.title(),lw=1)
axes[0].plot(times,accounted,label='Accounted total, work corrected',color='black',ls='--',lw=.9)
axes[0].set(xlabel='Time [microseconds]',ylabel='Energy [J]');axes[0].legend(fontsize=8)
axes[1].plot(times,velocity,color='#1a607e');axes[1].axhline(0,color='black',lw=.7)
axes[1].set(xlabel='Time [microseconds]',ylabel='Normal COM velocity from global P/M [m/s]')
for ax in axes:
    ax.grid(alpha=.2);ax.axvline(100,color='grey',ls=':',lw=.8)
fig.suptitle('Actual global history: energy transfer and wall-normal momentum',fontsize=11)
fig.savefig(destination/'phone-500us-energy-momentum.png',dpi=160);plt.close(fig);gc.collect()
guard('history_plot_complete')
spec=importlib.util.spec_from_file_location('bounded',ROOT/'work/expanded-drop/summarize-500us-bounded.py')
bounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounded)
frame=bounded.Frame(ROOT/stats[-1]['file'])
# Spatial illustration is deterministically thinned; scalar statistics use every solid.
selected=np.unique(np.r_[np.arange(0,frame.nelem,5),np.argmax(frame.array(frame.scalars['Von_Mises'],(frame.nelem,))),np.flatnonzero(frame.array(frame.scalars['Plastic_Strain'],(frame.nelem,))>0)])
coordinates=frame.array(frame.coords,(frame.nnode,3))
connectivity=frame.array(frame.connectivity,(frame.nelem,8),'>i4')
centers=np.empty((len(selected),3),dtype=np.float32)
for begin in range(0,len(selected),4096):
    cells=np.sort(connectivity[selected[begin:begin+4096]],axis=1)
    unique=np.ones(cells.shape,dtype=bool);unique[:,1:]=cells[:,1:]!=cells[:,:-1]
    centers[begin:begin+4096]=(coordinates[cells]*unique[:,:,None]).sum(axis=1)/unique.sum(axis=1)[:,None]
vm=frame.array(frame.scalars['Von_Mises'],(frame.nelem,))[selected].astype(np.float32)
epsp=frame.array(frame.scalars['Plastic_Strain'],(frame.nelem,))[selected].astype(np.float32)
strain_name=next(n for n in frame.tensors if n.startswith('Stra_') and 'Point111' in n)
strain=frame.array(frame.tensors[strain_name],(frame.nelem,6))[selected,0].astype(np.float32)
del coordinates,connectivity,cells,unique
frame.close();gc.collect();guard('before_spatial_plot')
fig,axes=plt.subplots(1,3,figsize=(12.5,4.6),layout='constrained')
for ax,values,title,cmap in zip(axes,[vm,strain,epsp],['Native cell von Mises [MPa]','Native total strain xx, IP111 [1]','Native cell EPSP [1]'],['magma','coolwarm','Reds']):
    order=np.argsort(np.abs(values));kwargs={}
    if title.startswith('Native total'):
        bound=max(float(np.abs(values).max()),1e-12);kwargs={'vmin':-bound,'vmax':bound}
    plot=ax.scatter(centers[order,0],centers[order,1],c=values[order],s=1.1,cmap=cmap,edgecolors='none',**kwargs)
    ax.set(xlabel='Global x [mm]',ylabel='Global y [mm]',title=title,aspect='equal');ax.set_facecolor('#e9edf0')
    fig.colorbar(plot,ax=ax,shrink=.75)
fig.suptitle(f"Actual FEM snapshot {stats[-1]['actual_time_s']*1e6:.3f} microseconds\nCell centers, XY projection; every fifth solid plus all plastic cells and the stress maximum",fontsize=11)
fig.savefig(destination/'phone-500us-native-fields.png',dpi=150);plt.close(fig);gc.collect()
measurement=guard('all_plots_complete')
(destination/'plot-resource-measurement.json').write_text(json.dumps(measurement,indent=2))
print(json.dumps({'status':'plots_completed','resource':measurement}))
