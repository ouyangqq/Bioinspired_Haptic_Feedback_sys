# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:35:46 2019
@author: qiangqiang ouyang
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import actuators_sensors as sa
import Receptors as rslib

argrithms=['BHF','LHF','PDHF','NHF']
titles=['(a)','(b)']

def plot_sc_curve(sc_res):
    disn=len(sc_res)
    fig =plt.figure(figsize=(7*disn,6))
    plt.subplots_adjust(hspace=0.3) 
    plt.subplots_adjust(wspace=0.2) 
    i=0
    argrithm=1 
    for scdata in sc_res:
        ax = fig.add_subplot(3,disn,disn+i-1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        plt.title(titles[i],fontsize=16)
        if(i==0):plt.ylabel('Pressure [N]')
        #st=np.where(scdata[0][:,0]==0)[0][-1]
        time=scdata[0][:,0]
        p=scdata[0][:,1]
        plt.plot(time,p,color='k')
        plt.yticks([0,0.4,0.8,1.2],fontsize=8)
        
        ax = fig.add_subplot(3,disn,2*disn+i-1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        time=scdata[1][:,0]
        for argrithm in range(3):
            ctrl_pneu=scdata[1][:,1+argrithm*2]
            plt.plot(time,ctrl_pneu,label=argrithms[2-argrithm])
        if(i==0):plt.ylabel('Pneumatic ')
        plt.yticks([0,20,40,60,80],fontsize=8)
        plt.legend(fontsize=8)
        
        ax = fig.add_subplot(3,disn,3*disn+i-1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        for argrithm in range(3):
            ctrl_vibro=scdata[1][:,2+argrithm*2]
            plt.plot(time,ctrl_vibro,label=argrithms[2-argrithm])
        if(i==0):plt.ylabel('Vibration')
        plt.xlabel('Time [s]')
        plt.yticks([0,20,40,60,80],fontsize=8)
        plt.legend(fontsize=8)
        i=i+1

'------------Sensor and control data from constructed stimuli-----------------'
simT=20
t= np.linspace(0,simT,int(simT/sa.dfT))
Pf=0.8
arti_Pd1=rslib.step_wave(t,0.05*simT,0.35*simT,500,-500,Pf)
arti_Pd1=arti_Pd1+rslib.step_wave(t,0.65*simT,0.90*simT,2000,-2000,Pf)
arti_Pd2=rslib.step_wave(t,0.05*simT,0.35*simT,2000,-2000,0.4)
arti_Pd2=arti_Pd2+rslib.step_wave(t,0.45*simT,0.8*simT,2000,-2000,Pf)

#arti_Pd1=rslib.step_wave(t,0.05*simT,0.3*simT,2000,-4000,Pf)
#arti_Pd1=arti_Pd1+rslib.step_wave(t,0.375*simT,0.625*simT,4000,-4000,Pf)
#arti_Pd1=arti_Pd1+rslib.step_wave(t,0.7*simT,0.95*simT,6000,-4000,Pf)
#arti_Pd2=rslib.step_wave(t,0.05*simT,0.25*simT,3000,-3000,Pf/2)
#arti_Pd2=arti_Pd2+rslib.step_wave(t,0.325*simT,0.525*simT,3000,-3000,Pf)
#arti_Pd2=arti_Pd2+rslib.step_wave(t,0.6*simT,0.8*simT,3000,-3000,2*Pf)

sensordata1=np.vstack([t,arti_Pd1]).T
sensordata2=np.vstack([t,arti_Pd2]).T
A1=sa.ctrlcurve_recreated_from_sensordata(0.1,sa.ssize+1,sensordata1)
A2=sa.ctrlcurve_recreated_from_sensordata(0.1,sa.ssize+1,sensordata2)
np.save('data/scdata_art_wave.npy',[A1,A2])


scres=np.load('data/scdata_art_wave.npy')
plot_sc_curve(scres)
plt.savefig('hfs_figs/ssdata_arti_stimuli.png',bbox_inches='tight', dpi=300)



'-------------Sensor and control data from sensor stimulu--------------------------'
filepath='Recorded_data\wavetest/'
files=os.listdir(filepath)
#buf=np.load(filepath+files[2])  
buf=np.load('data/scdata_testwave.npy')
sensordata,ctrldata=np.array(buf[0]),np.array(buf[1])
#sensordata[:,1]=sensordata[:,1]
A=sa.ctrlcurve_recreated_from_sensordata(0.1,sa.ssize,sensordata)
np.save('data/scdata_testwave.npy',A)


fig =plt.figure(figsize=(7,6))
i=0
argrithm=1
disn=1
A=np.load('data/scdata_testwave.npy')
for scdata in [A]:
    ax = fig.add_subplot(3,disn,1+i)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None') 
    if(i==0):plt.ylabel('Pressure [N]')
    #st=np.where(scdata[0][:,0]==0)[0][-1]
    time=scdata[0][:,0]
    p=scdata[0][:,1]
    plt.plot(time,p)
    ty=np.arange(0,2,0.1)
    Tss=[4.1,7.5,10.5,13.75]
    
    ts=np.arange(0,20,0.1)
    plt.plot(ts,0.15*np.ones(len(ts)),'k--',linewidth=1)
    
    for ts in range(4):
        tx=Tss[ts]*np.ones(len(ty))
        plt.plot(tx,ty,'g--',linewidth=1)
        #plt.plot(ftiproi[:,0]+imgw/3,-ftiproi[:,1]+imgh/2,'y-',linewidth=1)
        #plt.fill_between(ftiproi[:,0]+imgw/3,-ftiproi[:,1]+imgh/2,facecolor='y',alpha=0.5) 
        plt.annotate('$\mathrm{T}_{'+str(ts)+'}$'+':'+str(Tss[ts]), 
                     xy=(Tss[ts],0.5), xytext=(Tss[ts]+1,1.5),
                     arrowprops=dict(color='c',headwidth = 4,width = 0.1,shrink=0.05))
    plt.yticks([0,1,2],fontsize=10)
    
    ax = fig.add_subplot(3,disn,2+i)
    
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    time=scdata[1][:,0]
    for argrithm in range(3):
        ctrl_pneu=scdata[1][:,1+argrithm*2]
        plt.plot(time,ctrl_pneu,label=argrithms[2-argrithm])
    if(i==0):plt.ylabel('Pneumatic ')
    plt.yticks([0,20,40,60,80,100],fontsize=10)
    plt.legend(fontsize=8)
    
    ax = fig.add_subplot(3,disn,3+i)
    plt.yticks([0,20,40,60,80,100],fontsize=10)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    for argrithm in range(3):
        ctrl_vibro=scdata[1][:,2+argrithm*2]
        plt.plot(time,ctrl_vibro,label=argrithms[2-argrithm])
    if(i==0):plt.ylabel('Vibration')
    plt.xlabel('Time [s]')
    plt.legend(fontsize=8)
    i=i+1

plt.savefig('hfs_figs/ssdata_some_stimulu.png',bbox_inches='tight', dpi=300) 