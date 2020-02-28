# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:35:46 2019

@author: qiangqiang ouyang
"""
import numpy as np
import matplotlib.pyplot as plt
import os
'''
folder = os.getcwd()[:-4] + 'new_folder\\test\\'
#获取此py文件路径，在此路径选创建在new_folder文件夹中的test文件夹
if not os.path.exists(folder):
    os.makedirs(folder)
folder = os.getcwd()[:-4] + 'new_folder\\test\\'
'''    
filepath='D:\\backup_research\My_workshop\model_haptic_sensory_sys\Bioinspired_Haptic_Feedback_sys\Training_course/'


files=os.listdir(filepath)
#effect=res.split('_')
#for i in range(3):   


buf=np.load(filepath+files[13])
   
A1,A2=np.array(buf[0]),np.array(buf[1])
fig =plt.figure(figsize=(12,6)) 




 
selects=[11,12,13]    
for s in range(3):
    buf=np.load(filepath+files[selects[s]])
    A1,A2=np.array(buf[0]),np.array(buf[1])
    
    #td=A2[1,0]-A2[0,0]
    pd=np.zeros([len(A2),1])
    for i in range(len(A2)):
        #sel=(A1[:,0]>A2[i,0])&(A1[:,0]<A2[i,0]+td)
        tmp=np.abs(A1[:,0]-A2[i,0])
        sel=tmp==np.min(tmp)
        pd[i]=np.average(A1[sel,1])
    
    ax = fig.add_subplot(3,5,1+5*s)
    ax.plot(A1[:,0],A1[:,1])
    #ax.set_xticks([0,4,8,12])
    ax.set_xlabel('time [s]')
    
    ax = fig.add_subplot(3,5,2+5*s)
    ax.plot(A2[:,0],A2[:,1])
    #ax.set_xticks([0,4,8,12])
    ax.set_xlabel('time [s]')
    
    ax = fig.add_subplot(3,5,3+5*s)
    ax.plot(A2[:,0],A2[:,2])
    #ax.set_xticks([0,4,8,12])
    ax.set_xlabel('time [s]')
    
    ax = fig.add_subplot(3,5,4+5*s)
    ax.scatter(pd,A2[:,1])
    ax.set_xlabel('Presure [N]')
    ax = fig.add_subplot(3,5,5+5*s)
    ax.scatter(pd,A2[:,2])
    ax.set_xlabel('Presure [N]')