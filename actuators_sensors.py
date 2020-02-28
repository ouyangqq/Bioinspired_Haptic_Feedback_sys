# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:03:15 2019
@author: qiangqiang ouyang
"""
import sys
#sys.path.append(r"../Skin_mechanics")
import os
#import skinareas as skas 
import time as ct
import numpy as np
#sys.path.append(r"../CAPR")
import Receptors as rslib
import img_to_eqstimuli as imeqst
import simset as mysim
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

Ttype_buf=['SA1','RA1','PC']
simT=0.2
modelling_freq=500

pbuf=np.load('data/loc_pos_buf_fingertip.npy')  
tsensors=[]
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=simT,sample_rate=modelling_freq,Density=pbuf[tp][2],roi=mysim.fingertiproi) 
    print(Ttype_buf[tp]+': '+str(tsensor.Rm))
    tsensors.append(tsensor) 

poisson_v=0.4
E_ym=50*1e3  #pa,Young modulus 
Cm=(1-poisson_v**2)/(2*E_ym)   
count1=0
#x0=0.38
px0=-0.0381
sampled_T=simT
dfT=0.018 #ct.time()-C1 #data updating period
multiplier=(sampled_T-dfT)/simT

sensor_up=2
sensor_down=px0
pneumatic_down=36
pneumatic_up=99
vibro_down=24
vibro_up=99

paras=np.load('data/act_paras.npy').item()

def func(x,a,wbc):  
    return a*(1+wbc)**(x-1)
    #return a*(np.log(x)/np.log(b))
    #return a*log(x)+b
def presure_from_AD(ad,x0):
    y=7e-8*ad**3+2e-5*ad**2+0.0082*ad+x0 #x0=0.0381
    return y

def psy_fucnt_vibro(perceived_intensity):
    val=paras['v_c0']*((1+paras['v_w'])**(perceived_intensity))
    return (val>vibro_up)*vibro_up+val*(val<=vibro_up)
    
def psy_fucnt_normalforce(perceived_intensity):
    val=paras['p_c0']*((1+paras['p_w'])**(perceived_intensity))
    return (val>pneumatic_up)*pneumatic_up+val*(val<=pneumatic_up)

rgbuf=presure_from_AD(np.array([200,250,300,700])/4,px0)

def current_controlling(buf):
    avg=np.average(buf)
    if((avg<rgbuf[0])):res=[0,0] 
    elif((avg>=rgbuf[0])&(avg<rgbuf[1])): res=  [30,00]
    elif((avg>=rgbuf[1])&(avg<rgbuf[2])): res=  [50,30]
    elif((avg>=rgbuf[2])&(avg<rgbuf[3])): res=  [80,70]
    elif((avg>rgbuf[3])):  res= [99,99]
    return res

def linear_controlling(buf):
    vibro_c=(buf[:]-sensor_down)/(sensor_up-sensor_down)*(vibro_up-vibro_down)+vibro_down
    pneumatic_c=(buf[:]-sensor_down)/(sensor_up-sensor_down)*(pneumatic_up-pneumatic_down)+pneumatic_down
    C=[np.average(pneumatic_c),np.average(vibro_c)]
    if(C[0]<0): C[0]=0
    if(C[1]<0): C[1]=0
    return [C[0],C[1]]
    
rad=6
baseh=0.0
probeh=5
[pimage,eqs]=mysim.constructing_probe_stimuli(np.array([[0,0,rad,probeh]]))
vibro_EEPS=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0]+baseh,pimage[1],pimage[2],mysim.skinroi[0])

rad=2
[pimage,eqs]=mysim.constructing_probe_stimuli(np.array([[0,0,rad,probeh]]))
pneu_EEPS=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0]+baseh,pimage[1],pimage[2],mysim.skinroi[0])

rad=4
[pimage,eqs]=mysim.constructing_probe_stimuli(np.array([[0,0,rad,probeh]]))
sensor_EEPS=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0]+baseh,pimage[1],pimage[2],mysim.skinroi[0])

SAEEPS=[sensor_EEPS,vibro_EEPS,pneu_EEPS]
np.save('data/SAEEPS.npy',SAEEPS)


p=[sensor_EEPS[0][0]/2,sensor_EEPS[0][1]/2] #applied site
ips=[np.hstack([p[0]*np.ones([tsensor.t.size,1]),
                p[1]*np.ones([tsensor.t.size,1]),
                np.zeros([tsensor.t.size,1])]),'Pressure']

    
[Kpv,Kpf]=np.load('data/act_Kps.npy')
#Kpv,Kpf=1/8,1/8

def predict_haptic_perception_using_CAPR(buf):
    #load data to input buf for model of afferent population responses1)
    #pds=rslib.butterworth_filter(2,buf,20,'low',modelling_freq)
    ips[0][:,2]=buf
    
    #np.array(buf)#(-sensor_down)/len(stimuli_dots[0])
    #for t in range(len(buf)):
    #    stimuli_dots[t][:,3]=(buf[t]-sensor_down)/len(stimuli_dots[0])
    'Runing the model of afferent population responses'
    Afr=[0,0,0]
    Frates=[]
    for tp in range(len(Ttype_buf)):
        tsensors[tp].population_simulate(EEQS=sensor_EEPS[0:2],Ips=ips,acquire_spikes=False,disinf=False,noise=0)
        tmp=tsensors[tp].Va==0.04
        Frates.append(np.sum(tmp,1)/simT)
        
        #tmp=np.average(tsensors[tp].Vnf*tsensors[tp].Kf/tsensors[tp].VL,1)
        #Frates.append(tmp)
        #tmp.append([tsensor_buf[tp].Vc,tsensor_buf[tp].Vf,tsensor_buf[tp].Va])  
        'Computing average firing rate for each afferent'
        sel=Frates[tp]>5 
        if(np.sum(sel)>0):
            Afr[tp]=np.average(Frates[tp][sel])
        
        if(tp==0):print('SA1-bio:',Afr[tp])
    
    'Computing rating of haptic perception'
    Pvibro_intensity=(0.29*Afr[0]+0.36*Afr[1]+0.46*Afr[2])*Kpv/10
    Pforce_intensity=(Afr[0])*Kpf/10
    
    return Pforce_intensity, Pvibro_intensity,
    
def bio_inspired_controlling(buf):
    Ipf,Ipv=predict_haptic_perception_using_CAPR(buf)
    return  [psy_fucnt_normalforce(Ipf),psy_fucnt_vibro(Ipv)]



ssize=int(sampled_T/dfT)
def ctrlcurve_recreated_from_sensordata(cdt,ssize,buf):
    'Loading detected data'
    sdt=(buf[-1,0]-buf[0,0])/len(buf)
    space=int(cdt/sdt)
    sampled_buf=np.zeros([ssize,2])
    i=0
    ctrl_buf=[]
    pds=np.zeros(tsensors[0].t.size)# detected pressure data buf
    while(1):
        sampled_buf[:,:]=buf[i*space:i*space+ssize,:]
        tp=(sampled_buf[-1,0]-sampled_buf[0,0])/multiplier
        if (tp>simT):
            f = interp1d(sampled_buf[:,0]/multiplier-sampled_buf[0,0]/multiplier, sampled_buf[:,1])
            pds[:]=f(tsensors[0].t)#
            ctrl1=np.array(current_controlling(pds))
            ctrl2=np.array(linear_controlling(pds))
            ctrl3=np.array(bio_inspired_controlling(pds))
            timestaple=buf[i*space,0]
            cc=np.hstack([timestaple,ctrl1,ctrl2,ctrl3])
            ctrl_buf.append(cc)
        i=i+1
        if(i*space+ssize>len(buf)): break
    return [buf,np.array(ctrl_buf)]