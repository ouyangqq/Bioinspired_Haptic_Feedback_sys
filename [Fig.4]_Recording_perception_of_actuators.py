# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:10:56 2019

@author: qiangqiang ouyang
"""

import os 
import sys
import ultils as alt
import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
import simset as mysim 
from PIL import Image
import img_to_eqstimuli as imeqst
import actuators_sensors as sa
#matplotlib.use('TkAgg')
# -*- coding: utf-8 -*-

width=80#mm
height=20#mm


shift=1
speed=40
pf=30*9.8*1e-3 #pressing force (35 g)
simT=0.3
simdt=0.001
Ttype_buf=['SA1','RA1','PC']
tsensors=[]

pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=simT,sample_rate=1/simdt,Density=pbuf[tp][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor)


Kpsi=0.454*9.8/(25.4)**2
Mg=0.001 #0.8g
APdata=np.round(np.loadtxt('data/txtdata/AD_P.txt'),2)

psycho_act_data=np.load('data/psycho_data_actuators.npy')
actouput_Cs=np.load('data/actouput_stlevel_func_coffecients.npy')
Dis,rp=psycho_act_data[1][:,0],2

def actouput_stlevel_func(x):  # stlevel: 0~100
    vibro_acc=actouput_Cs[0][0]*x**2+actouput_Cs[0][1]*x+actouput_Cs[0][2]
    vibro_f=actouput_Cs[1][0]*x**2+actouput_Cs[1][1]*x+actouput_Cs[1][2]
    pneumatic_PSI=(x>actouput_Cs[2][2])*(actouput_Cs[2][0]*x+actouput_Cs[2][1])+(x<=actouput_Cs[2][2])*0
    return [vibro_acc,vibro_f,pneumatic_PSI]
def Di_to_Cp(x):  # stlevel: 0~100
    return (1e-3)**2*x*rp/sa.Cm/(rp**2*np.pi*Kpsi*0.2)+actouput_Cs[2][2]

Cps=Di_to_Cp(Dis)
A=actouput_stlevel_func(psycho_act_data[0][:,0])[0:2]
A[0]=A[0]*Mg
B=actouput_stlevel_func(Di_to_Cp(Dis))[2]*Kpsi*np.pi*rp**2/2
np.save('data/stimuli_sets.npy',[np.mat(A).T*np.array([[0,1],[1,0]]),B])

#img1 =Image.open('saved_figs/brick_60-15.jpg')
#img2 =Image.open('saved_figs/marble_60-15.jpg')
#img3 =Image.open('saved_figs/school_log_60-15.jpg')
buf=np.load('data/stimuli_sets.npy')
#vlevels=np.array([[20,0.2],[30,0.32],[40,0.32],[75,0.4],[120,0.7],[150,1],[162,1.2]]) #vibro [freq, amp]
#plevels=np.array([1,2,3,4,5,6,7])/5
vlevels=buf[0]
plevels=buf[1]

def farbricate_haptic_signals():
    noise=np.random.uniform(-0.000000,0.00000,tsensor.t.size)
    buf1=[]
    buf2=[]
    for i in range(len(vlevels)):
        res=rslib.sin_wave(tsensor.t,2*np.pi*vlevels[i,0],vlevels[i,1])+noise
        buf1.append(res)
    
    
    for i in range(len(plevels)):
        res=np.ones(tsensor.t.size)*plevels[i]+noise/6
        #res=rslib.step_wave_any(tsensor.t,0.02,0.35,20,-20,plevels[i])+noise/6
        buf2.append(res)
        
    buf3=0.6*1e3*rslib.step_wave(tsensor.t,0.02,0.15,50,-40,1*1e-3)+rslib.sin_wave(tsensor.t,2*np.pi*100,0.05)+2*noise
    return buf1,buf2,buf3



vibro_ps_dfs,pneu_ps_dfs,sensor_ps=farbricate_haptic_signals()

names=['sensor_EEPS','vibro_EEPS','pneu_EEPS']
def plot_Act_FP_signals_and_eqstimuli():
    buf=vibro_ps_dfs
    plt.figure(figsize=(1.8,5))
    plt.subplots_adjust(hspace=0.4)
    for i in range(len(buf)):
        ax=plt.subplot(len(buf),1,i+1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        plt.plot(tsensor.t[0:],buf[i][0:],'k',linewidth=0.5)
        plt.xticks([])
        plt.yticks([-0.01,0,0.01],fontsize=6)
        if(i==5): plt.ylabel("Force [N]",fontsize=14)
        if(i==len(buf)-1):
            plt.xticks([0,simT/2,simT],fontsize=8)
            plt.xlabel("Time [s]")
            #plt.ylabel("Presure [N]")
        #plt.legend(loc=1,fontsize=6)
        freq=np.uint16(vlevels[i,0])
        force=np.round(vlevels[i,1],3)
        plt.text(simT,0,str(freq)+' Hz'+'\n'+str(force)+' N',fontsize=7)
        #plt.legend(loc=1,fontsize=6,edgecolor='w')
    plt.savefig('hfs_figs/vibro_stimulus_wave.png',bbox_inches='tight', dpi=300)  
     
    plt.figure(figsize=(5,7))
    buf=pneu_ps_dfs
    ax=plt.subplot(1,1,1)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    for i in range(len(buf)):
        plt.plot(tsensor.t[0:],buf[i][0:],label="level"+str(i+1),linewidth=0.8)
   
    plt.yticks([0,0.1,0.2,0.3])
    plt.xticks([0,0.1,0.2])
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend(loc=1,ncol=3,fontsize=6)
    plt.savefig('hfs_figs/pneu_stimulus_wave.png',bbox_inches='tight', dpi=300)  

    
def predict_haptic_perception_using_CAPR(buf,EEPS,act='ballon'):
    p=[EEPS[0][0]/2,EEPS[0][1]/2] #applied site
    ips=[np.hstack([p[0]*np.ones([tsensor.t.size,1]),
                    p[1]*np.ones([tsensor.t.size,1]),
                         np.zeros([tsensor.t.size,1])]),'Pressure']
    Kpv,Kpf=1,1
    #load data to input buf for model of afferent population responses1)
    #pds=rslib.butterworth_filter(2,buf-np.average(buf),10,'low',sa.modelling_freq)
    ips[0][:,2]=buf
    #print('bio:',buf[10])
    #np.array(buf)#(-sensor_down)/len(stimuli_dots[0])
    #for t in range(len(buf)):
    #    stimuli_dots[t][:,3]=(buf[t]-sensor_down)/len(stimuli_dots[0])
    'Runing the model of afferent population responses'
    Afr=[0,0,0]
    Frates=[]
    for tp in range(len(Ttype_buf)):
        tsensors[tp].population_simulate(EEQS=EEPS,Ips=ips,acquire_spikes=False,disinf=False,noise=0)
        tmp=tsensors[tp].Va==0.04
        Frates.append(np.sum(tmp,1)/simT)
        #tmp=np.average(tsensors[tp].Vnf*tsensors[tp].Kf/tsensors[tp].VL,1)
        #Frates.append(tmp)
        #tmp.append([tsensor_buf[tp].Vc,tsensor_buf[tp].Vf,tsensor_buf[tp].Va])  
        'Computing average firing rate for each afferent'
        #sel=tsensors[tp].points_mapping_entrys(np.array([[0,0]]))[0]
        sel=Frates[tp]>0.02*tsensors[tp].maxfr
        if(np.sum(sel)>0):
            Afr[tp]=np.average(Frates[tp][sel])
            print(Ttype_buf[tp],Afr[tp])
    
    'Computing rating of haptic perception'
    if(act=='vibro'):Pintensity=(0.29*Afr[0]+0.36*Afr[1]+0.46*Afr[2])*Kpv/10
    elif(act=='ballon'): Pintensity=Afr[0]*Kpf/10

    '''
    if(act=='vibro'):
        Pintensity=(0.29*np.average(Frates[0][Frates[0]>0])+
                    0.36*np.average(Frates[1][Frates[1]>0])+
                    0.46*np.average(Frates[2][Frates[2]>0]))*Kpv
    '''
    
    
    return Pintensity,  

plot_Act_FP_signals_and_eqstimuli()


tmp=[]
for i in range(len(vibro_ps_dfs)):
    res=predict_haptic_perception_using_CAPR(vibro_ps_dfs[i],sa.vibro_EEPS[0:2],act='vibro')
    tmp.append(res)
A_vibro_perceptions=np.array(tmp)

tmp=[]
for i in range(len(pneu_ps_dfs)):
    res=predict_haptic_perception_using_CAPR(pneu_ps_dfs[i],sa.pneu_EEPS[0:2],act='ballon')
    tmp.append(res)
A_ballon_perceptions=np.array(tmp)

np.save('data/act_perception_res.npy',[A_vibro_perceptions,A_ballon_perceptions])
plt.figure(figsize=(6,5))
plt.plot(A_vibro_perceptions)
plt.plot(A_ballon_perceptions)