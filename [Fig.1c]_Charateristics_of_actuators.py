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

#APdata=np.round(np.loadtxt('data/txtdata/AD_P.txt'),2)
#psycho_act_data=np.load('data/psycho_data_actuators.npy')

A=np.round(np.loadtxt('data/txtdata/coin_motor_308_freq.txt'),2)
B=np.round(np.loadtxt('data/txtdata/coin_motor_308_amp.txt'),2)
vibro_buf=np.hstack([A[:,0:1],A[:,1:2],B[:,1:2]])


fig =plt.figure(figsize=(6.5,3))
plt.subplots_adjust(wspace=0.3) 
fitting_paras=[]


#Yvibro=np.log(Pdata[:,0]/15)/np.log(1+0.31)+1
#Ypneu=np.log(Pdata[:,0]/11)/np.log(1+0.37)+1

ax = fig.add_subplot(1,2,2)
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
plt.title('Vibration motor',fontsize=14)  # Pressing force
plt.xlabel('Control amount (%)')


x,y=vibro_buf[:,0],vibro_buf[:,2]
fit=alt.ploy2_curve_fit(x,y)
fitting_paras.append(np.array(fit[3][0:3]))
#A='$\mathrm{ddd}^{ddd}$'
Ff='y='+'$\mathrm{'+"{0:.6f}".format(fit[3][0])+'x}^{2}+$'+\
        '$\mathrm{'+"{0:.2f}".format(fit[3][1])+'x}$'+\
        '$\mathrm{'+"{0:.2f}".format(fit[3][2])+'}$'

plt.plot(x,y, 'kd',markerfacecolor='none',markersize=6)
plt.plot(fit[0],fit[1],'-',color='b', label='Accleration\n'+Ff)

plt.xticks([0,20,40,60,60,80,100],fontsize=8)
plt.yticks([0,2,4,6,8,10,12,14,16],fontsize=8)

plt.ylabel('Acceleration ($\mathrm{m/s}^{2}$)')

plt.legend(loc=1,fontsize=8)
ax1=ax.twinx() 
ax1.spines['top'].set_color('None')
x,y=vibro_buf[:,0],vibro_buf[:,1]
fit=alt.ploy2_curve_fit(x,y)
fitting_paras.append(np.array(fit[3][0:3]))
#A='$\mathrm{ddd}^{ddd}$'
Ff='y='+'$\mathrm{'+"{0:.4f}".format(fit[3][0])+'x}^{2}+$'+\
        '$\mathrm{'+"{0:.2f}".format(fit[3][1])+'x}$'+\
        '$\mathrm{'+"{0:.2f}".format(fit[3][2])+'}$'

plt.plot(x,y,'ko',markerfacecolor='none',markersize=6)
plt.plot(fit[0],fit[1],'-',color='r', label='Frequency\n'+Ff)
#plt.xticks([0,20,40,60,60,80,100],fontsize=8)
plt.yticks([0,50,100,150,200,250,300,350,400],fontsize=8)
plt.ylabel('Frequency (Hz) ')
plt.legend(loc=(0.02,0.65),fontsize=8)


ax = fig.add_subplot(1,2,1)
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')
plt.title('Pneumatic ballon',fontsize=14)  # Pressing force
plt.xlabel('Control amount (%)')
plt.ylabel('pounds per $\mathrm{inch}^{2}$ (PSI)')



PSIs=np.array([9,11,14,20,24])

x=PSIs/25*100
y=PSIs-9
fit=alt.linear_curve_fit(x,y)
fit[3].append(x[0])
fitting_paras.append(np.array(fit[3]))
#A='$\mathrm{ddd}^{ddd}$'
Ff='y='+'$\mathrm{'+"{0:.2f}".format(fit[3][0])+'\dot x'+"{0:.2f}".format(fit[3][1])+' }$  x>'+"{0:.0f}".format(x[0])+' \ny=0                    x<'+"{0:.0f}".format(x[0])

plt.plot(x,y, 'ko',markerfacecolor='none',marker='o',markersize=6)


x0s=np.arange(0,x[0],1)
y0s=np.zeros(len(x0s))
plt.plot(np.hstack([x0s,fit[0]]),np.hstack([y0s,fit[1]]), color='b',label=Ff)
plt.plot(np.ones(10)*x[0],np.linspace(0,7,10), '--',color='gray')
plt.text(x[0]-10,7.5,'$\mathrm{C}_{p0}$='+str(x[0]),fontsize=12)
#R2v="{0:.3f}".format(alt.R2(fit[2],y))
plt.legend(loc=2,fontsize=8)
   
plt.xticks([0,20,40,60,60,80,100],fontsize=8)
plt.yticks([0,5,10,15],fontsize=8)

np.save('data/actouput_stlevel_func_coffecients',fitting_paras)


plt.savefig('hfs_figs/characterstics_actuators.png',bbox_inches='tight', dpi=300) 