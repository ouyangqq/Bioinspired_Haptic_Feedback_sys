# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:41:02 2019

@author: qiang
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import actuators_sensors as sa
import Receptors as rslib
import simset as mysim
import ultils as alt
import math

Kpsi=0.454*9.8/(25.4)**2
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



'''
x=psycho_act_data[0][:,0]
y=psycho_act_data[0][:,1]
fit1=alt.log1_curve_fiting(x,y)

x=Cps
y=psycho_act_data[1][:,1]
fit2=alt.log1_curve_fiting(x,y)
'''


Pdata=[psycho_act_data[0][:,0],
       np.load('data/act_perception_res.npy')[0][:,0],
       Cps,
       np.load('data/act_perception_res.npy')[1][:,0]]


fig =plt.figure(figsize=(12,3.5))
plt.subplots_adjust(wspace=0.4) 
lw=0.5
#Yvibro=np.log(Pdata[:,0]/15)/np.log(1+0.31)+1
#Ypneu=np.log(Pdata[:,0]/11)/np.log(1+0.37)+1

ax = fig.add_subplot(1,3,1)
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
plt.title('(a) Vibration motor',fontsize=14)  # Pressing force
#plt.xlabel('Stimulus level (%)')
plt.xlabel('Controlling amount ($\mathrm{C}_{v}$)')
plt.ylabel('Magnitude of perceived intensity')

#plt.scatter(psycho_act_data[0][:,0],psycho_act_data[0][:,1],
#            color='w',edgecolors='k',marker='o',s=20,
#           label='Psycho')

plt.plot(psycho_act_data[0][:,0],psycho_act_data[0][:,1],
         '-',color='k', linewidth=lw,
         marker='o',markerfacecolor='none',markersize=6,label='Psycho')

Iv1=np.average(psycho_act_data[0][:,1])
Iv0=np.average(Pdata[1])
Kpv=Iv1/Iv0
x=psycho_act_data[0][:,1]
y=Pdata[1]*Kpv
fit=alt.linear_curve_fit(x,y)
R2v="{0:.2f}".format(alt.R2(fit[2],y))
plt.plot(Pdata[0],Pdata[1],'-',color='red',marker='.',linewidth=lw,label='$\mathrm{K}_{pv}$=1') 

plt.plot(Pdata[0],Pdata[1]*Kpv,'-',color='b',marker='.',markersize=6,linewidth=lw,
         label='$\mathrm{K}_{pv}$='+str(round(Kpv,2))+', $\mathrm{R}^{2}$='+R2v)



#plt.text(25,Iv1+0.3,'$\mathrm{I}_{v}$ ='+str(round(Iv1,2)))
#plt.text(70,Iv0+0.3,'$\mathrm{I}_{pv}$($\mathrm{K}_{pv}$=1)='+str(round(Iv0,2)),color='red')
#plt.plot(np.linspace(0,100,50),np.ones(50)*Iv1,'--',color='k')
#plt.plot(np.linspace(0,100,50),np.ones(50)*Iv0,'--',color='red')


plt.xlim((20,100))   
plt.yticks([0,1,2,3,4,5,6,7,8,9],fontsize=8)
plt.xticks([20,40,60,60,80,100],fontsize=8)
plt.legend(fontsize=9)



#ax = fig.add_subplot(2,3,2)
stx,sty,xl=0.39,0.22,0.225
ax1=plt.axes([stx,sty-0.12,xl, 0.05])
#plt.subplots_adjust(hspace=0) 
ax1.spines['top'].set_color('None')
ax1.spines['right'].set_color('None')
ax1.spines['left'].set_color('None')
ax1.set_yticks([])
#ax1.xaxis.set_ticks_position('top')
#ax1.xaxis.set_label_position('top')
#plt.xticks([0,0.5,1,1.5],fontsize=8)
plt.xlim((psycho_act_data[1][0,0],psycho_act_data[1][-1,0]))
##plt.xlim(0,1.6,0.2)
plt.xticks(np.round(np.arange(psycho_act_data[1][0,0],psycho_act_data[1][-1,0]+0.3,0.3),1),fontsize=8)
ax1.set_xlabel('Indentation (mm)',fontsize=8,va='top')


ax2=plt.axes([stx-0.007,sty,xl+0.007, 0.65])
ax2.spines['top'].set_color('None')
ax2.spines['right'].set_color('None')
plt.title('(b) Pneumatic balloon',fontsize=14)  # Pressing force
plt.ylabel('Magnitude of perceived intensity')
#plt.scatter(Di_to_Cp(psycho_act_data[1][:,0]),psycho_act_data[1][:,1],
#            color='w',edgecolors='k',marker='o',s=20,
#            label='Psycho')

plt.plot(Di_to_Cp(psycho_act_data[1][:,0]),psycho_act_data[1][:,1],
         '-',color='k',linewidth=lw,
         marker='o',markerfacecolor='none',markersize=6,label='Psycho')



If1=np.average(psycho_act_data[1][:,1])
If0=np.average(Pdata[3])
Kpf=If1/If0

x=psycho_act_data[1][:,1]
y=Pdata[3]*Kpf
fit=alt.linear_curve_fit(x,y)
R2v="{0:.2f}".format(alt.R2(fit[2],y))

ax2.plot(Pdata[2],Pdata[3],'-',color='red',
         linewidth=lw,marker='.',label='$\mathrm{K}_{pf}$=1') 

ax2.plot(Pdata[2],Pdata[3]*Kpf,'-',color='b',linewidth=lw,
         marker='.',markersize=5,
         label='$\mathrm{K}_{pf}$='+str(round(Kpf,2))+', $\mathrm{R}^{2}$='+R2v) 

plt.xlim((Cps[0]-1,Cps[-1]+1))
plt.xticks(np.round(np.arange(Cps[0],Cps[-1]+4,4),1),fontsize=8)
plt.yticks([0,1,2,3,4,5,6,7,8,9],fontsize=8)
#plt.xlabel('Stimulus level (%)')

#plt.plot(np.linspace(Cps[0],100,50),np.ones(50)*If1,'--',color='k')
#plt.plot(np.linspace(0,100,50),np.ones(50)*np.average(Pdata[3]),'--',color='red')
#plt.text(Cps[0]+1,If1+0.2,'$\mathrm{I}_{f}$ ='+str(round(If1,2)))
#plt.text((Cps[0]+Cps[-1])*1.05/2,If0-0.8,'$\mathrm{I}_{pf}$($\mathrm{K}_{pf}$=1)='+str(round(If0,2)),color='red')

plt.xlabel('Controlling amount ($\mathrm{C}_{p}$)',fontsize=8,va='top')
#ax1 = fig.add_subplot(10,3,29)
plt.legend(loc=2,fontsize=9)


#plt.setp(ax1.get_xticklabels(), visible=True)
ax = fig.add_subplot(1,3,3)
labels=['Vibro','Pneumatic']
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
plt.title('(c) Inverse function of perception',fontsize=12)  # Pressing force
plt.xlabel('Magnitude of perceived intensity (I)',fontsize=10)
plt.ylabel('Controlling amount ($\mathrm{C}_{}$)',fontsize=10)
labels=['Vibro','Pneumatic']


Kps=[Kpv,Kpf]
webers=[]
psychodata=[[psycho_act_data[0][:,0],psycho_act_data[0][:,1]],
[Di_to_Cp(psycho_act_data[1][:,0]),psycho_act_data[1][:,1]]]
clos=['m','g']
for ch in range(2):
    x=psychodata[ch][1]
    y=psychodata[ch][0]
    #y=Pdata[2*ch]
    #x=Pdata[2*ch+1]*Kps[ch]
    fit=alt.exp_curve_fiting(x,y)
    webers.append([fit[3][0],fit[3][1]])
    Ff='C=$\mathrm{'+"{0:.3f}".format(fit[3][0])+'*(1+'+"{0:.3f}".format(fit[3][1])+')}^{I}$'
    #R2v="{0:.3f}".format(alt.R2(fit[2],y))
    #plt.plot(x,y,marker=mysim.markers[ch],markersize=4.5,markerfacecolor='none',
    #             color=mysim.colors[ch],linewidth=1,
    #            label=labels[ch])
    #plt.scatter(x,y,color='w',edgecolors=mysim.colors[ch],marker=mysim.markers[ch+1],s=10)
    plt.plot(fit[0],fit[1],'-',color=clos[ch],linewidth=2,
             label=labels[ch]+'\n'+Ff)
    
    #X=np.arange(0,8,0.01)
    #if(ch==0):Y=15*((1+0.31)**(X-1))
    #if(ch==1):Y=11*((1+0.37)**(X-1))
    #plt.plot(X,Y,color=mysim.colors[ch])   
plt.xticks([0,1,2,3,4,5,6,7,8,9],fontsize=8)
plt.yticks([20,40,60,60,80,100,120],fontsize=8)
plt.legend(fontsize=8)

paras={'Kpv':Kpv,'Kpf':Kpf, 
       'v_c0': webers[0][0], 
       'v_w': webers[0][1],
       'p_c0': webers[1][0], 
       'p_w': webers[1][1]} 

np.save('data/act_paras.npy',paras)
#np.save('data/act_Kps.npy',Kps)
plt.savefig('hfs_figs/Inverse_fitting.png',bbox_inches='tight', dpi=300) 