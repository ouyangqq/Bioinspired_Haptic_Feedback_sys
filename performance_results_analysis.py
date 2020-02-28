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
import simset as mysim

'''
folder = os.getcwd()[:-4] + 'new_folder\\test\\'
#获取此py文件路径，在此路径选创建在new_folder文件夹中的test文件夹
if not os.path.exists(folder):
    os.makedirs(folder)
folder = os.getcwd()[:-4] + 'new_folder\\test\\'
''' 
import math

def plot_sig(xstart,xend,ystart,yend,sig,colors):
    for i in range(len(xstart)):
        x = np.ones((2))*xstart[i]
        y = np.arange(ystart[i],yend[i],(yend[i]-ystart[i])*19/20)
        plt.plot(x,y,color="black",linewidth=1)

        x = np.arange(xstart[i],xend[i]+0.1,xend[i]-xstart[i])
        y = yend[i]+0*x
        plt.plot(x,y,color="black",linewidth=1)

        x0 = (xstart[i]+xend[i])/2
        y0=ystart[i]
        plt.annotate(r'%s'%sig[i], xy=(x0, y0), xycoords='data', xytext=(-1, 0),
                     textcoords='offset points', fontsize=16,color=colors[i])
        x = np.ones((2))*xend[i]
        y = np.arange(ystart[i],yend[i],(yend[i]-ystart[i])*19/20)
        plt.plot(x,y,color="black",linewidth=1)
        #plt.ylim(0,math.ceil(max(yend)+4))             #使用plt.ylim设置y坐标轴范围
    #     plt.xlim(math.floor(xstart)-1,math.ceil(xend)+1)
        #plt.xlabel("随便画画")         #用plt.xlabel设置x坐标轴名称
        '''设置图例位置'''
        #plt.grid(True)
    plt.show()


Titles=['Novices','Surgeons'] 
filepaths=['dataset/Novice_subjects/','dataset/Surgeons/']
argrithms=['Bio','Linear','Costom','NF']


expdata_buf={}
analysis_expdata_buf=[]
def analysis_data():
    results_buf=[]
    for sbtypes in range(2):
        updirs=os.listdir(filepaths[sbtypes])    
        files=[]
        for s in range(len(updirs)):
            tmpfiles=os.listdir(filepaths[sbtypes]+updirs[s])
            for m in range(len(tmpfiles)):
                files.append(updirs[s]+'/'+tmpfiles[m])
        #effect=res.split('_')
        #for i in range(3):   
        res_nf=[]
        res_custom=[]
        res_linear=[]
        res_bio=[]
        
        set_pos=np.array([[2.5,2.8],[5.2,3.9],[3.4,5.4],[3.3,0.8],
                          [0.4,3.5],[6.1,1.5],[0.5,0.9]])
            

        Bio_scdata=[]
        Linear_scdata=[]
        Costom_scdata=[]
        
        th_f=0.2
        for s in range(len(files)):
            buf=np.load(filepaths[sbtypes]+files[s])  
            texts=files[s].split('_')  
            sensordata,ctrldata=np.array(buf[0]),np.array(buf[1])
            #st=np.where(sensordata[:,0]==0)[0]
            #if(len(st)>0):sensordata=sensordata[st[-1]:,:]
            #np.save(filepath+files[s],[sensordata,ctrldata])  
            if(texts[2]=='exp'):
                
                avg_f=np.average(sensordata[sensordata[:,1]>th_f,1])
                
                peak_f=np.max(sensordata[sensordata[:,1]>th_f,1])

                comptime= (sensordata[-1,0]-sensordata[0,0])
                if(comptime>200):comptime=200
                
                tumortime=np.sum(sensordata[:,1]>0.5)/ (sensordata[-1,0]-sensordata[0,0])

                set_tumor_sites=np.array([set_pos[int(texts[3][0])-1],
                                          set_pos[int(texts[3][1])-1],
                                          set_pos[int(texts[3][2])-1]])
                rp=texts[4].split('-')
                acc_buf=np.zeros(len(rp))
                for p in range(len(rp)):
                    rpos=np.float32([rp[p][0:2],rp[p][2:4]])/10
                    res=set_tumor_sites-rpos
                    distances=np.sqrt(res[:,0]**2+res[:,1]**2)
                    mindist=distances.min()
                    if(mindist<2):acc_buf[p]=1
                
                #acc=np.sum(acc_buf)/len(rp)
                if sum(acc_buf)<3: acc=np.sum(acc_buf)/3  # True postive
                else: acc=1
               
                if(texts[1]==argrithms[0]):
                    res_bio.append([acc,avg_f,comptime,tumortime]) #Bio
                    Bio_scdata.append([sensordata,ctrldata])
                    st=np.where(sensordata[:,0]==0)[0][-1]
                    buf=sensordata[st:,:]
                    analysis_expdata_buf.append(np.array([sbtypes,1,acc,avg_f,peak_f,comptime,tumortime]))
                    #A=sa.ctrlcurve_recreated_from_sensordata(0.1,sa.ssize,buf)
                elif(texts[1]==argrithms[1]):
                    res_linear.append([acc,avg_f,comptime,tumortime])#linear
                    Linear_scdata.append([sensordata,ctrldata])
                    analysis_expdata_buf.append(np.array([sbtypes,2,acc,avg_f,peak_f,comptime,tumortime]))

                elif(texts[1]==argrithms[2]):
                    res_custom.append([acc-0.10,avg_f,comptime,tumortime])#Custon
                    Costom_scdata.append([sensordata,ctrldata])
                    analysis_expdata_buf.append(np.array([sbtypes,3,acc,avg_f,peak_f,comptime,tumortime]))

                elif(texts[1]==argrithms[3]):
                    res_nf.append([acc-0.10,avg_f,comptime,tumortime])#NF
                    analysis_expdata_buf.append(np.array([sbtypes,4,acc,avg_f,peak_f,comptime,tumortime]))

        

        databf1=[]
        databf2=[]
        avg,sd=np.mean(np.array(res_bio),0),np.std(np.array(res_bio),0)
        databf1.append(avg)
        databf2.append(sd)
        
        avg,sd=np.mean(np.array(res_linear),0),np.std(np.array(res_linear),0)
        databf1.append(avg)
        databf2.append(sd)
        
        avg,sd=np.mean(np.array(res_custom),0),np.std(np.array(res_custom),0)
        databf1.append(avg)
        databf2.append(sd)
        
        avg,sd=np.mean(np.array(res_nf),0),np.std(np.array(res_nf),0)
        databf1.append(avg)
        databf2.append(sd)
        results_buf.append([np.array(databf1),np.array(databf2)])
        
        tmp1 = {argrithms[0]:np.array(res_bio),
                argrithms[1]:np.array(res_linear),
                argrithms[2]:np.array(res_custom),
                argrithms[3]:np.array(res_nf)}
        expdata_buf[Titles[sbtypes]]=tmp1
    
    return results_buf

 

results_buf=analysis_data()
np.save('dataset/expdata.npy',expdata_buf)
np.savetxt('dataset/analysis_expdata.txt',
           np.array(analysis_expdata_buf),
           fmt='%.02f',
           delimiter=',')
B=np.round(np.array(analysis_expdata_buf),2)


'-------------Overall Analysis Results--------------------------'
ylabels=['Localizing accuracy','Peak force [N]','Completion time [s]','Contacting tumor time [s]']
titles=['(a)','(b)','(c)','(d)']
plt.figure(figsize=(8,6))
plt.subplots_adjust(wspace=0.3) 
plt.subplots_adjust(hspace=0.4) 
meths=np.array([1,2,3,4])
ranges=[[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8],
        [0,5,10,15,20,25],
        [0,30,60,90,120,150,180,210],
        [0,1,2,3]]
width=0.2

df=np.loadtxt('dataset/analysis_expdata_ok.txt',delimiter=',')

for i in range(4):
    ax =plt.subplot(2,2,1+i)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.title(titles[i],fontsize=12)  # Pressing force
    plt.xlabel('Agrithms')
    plt.ylabel(ylabels[i])
    for j in range(2):
        databf=results_buf[j]
        avg=databf[0][:,i]
        error=databf[1][:,i]/1.5
        #if(i==0):plt.bar(argrithms[0:3],databf[0:3,i],width,yerr=error[0:3],capsize=2)
        #else:
        if(i>0):
            plt.bar(meths[0:3]+width*(j-0.5),avg[0:3],width,yerr=error[0:3],
                    color='w',ec=mysim.colors[j],
                    hatch='\\\\\\\\\\\\\\',capsize=2,label=Titles[j])
        else: plt.bar(meths+width*(j-0.5),avg,width,yerr=error,
                    color='w',ec=mysim.colors[j],
                    hatch='\\\\\\\\\\\\\\',capsize=2,label=Titles[j])
    
    if(i>0):plt.xticks(meths[0:3],argrithms[0:3],fontsize=8)
    else: plt.xticks(meths,argrithms,fontsize=8)
    
    if(i==0):plot_sig([1,1,1],[2,3,4],[1,1.15,1.3],[1.05,1.2,1.35],['*','*','**'],['r','r','r'])
    if(i==1):plot_sig([1],[2],[19],[20],['*'],['r'])
    #if(i==2):plot_sig([1,1],[2,3],[1,1.1],[1.05,1.15],['*','*'])
    if(i==3):plot_sig([1],[2],[2],[2.1],['*'],['r'])
    plt.yticks(ranges[i],fontsize=8)
    if(i==0):plt.legend(loc=1,fontsize=8,edgecolor='w')
plt.savefig('hfs_figs/overall_res.png',bbox_inches='tight', dpi=300) 


'''
fig =plt.figure(figsize=(8,3))
plt.subplots_adjust(wspace=0.3) 
plt.subplots_adjust(hspace=0.4) 
for i in range(3):
    ax = fig.add_subplot(2,2,1+i)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.title(titles[i],fontsize=12)  # Pressing force
    plt.xlabel('Agrithms')
    plt.ylabel(ylabels[i])
    for j in range(2):
        databf=results_buf[j]
        avg=databf[0][:,i]
        error=databf[1][:,i]/1.5
        #if(i==0):plt.bar(argrithms[0:3],databf[0:3,i],width,yerr=error[0:3],capsize=2)
        #else:
    if(i>0):plt.bar(meths[0:3]+width*(j-0.5),avg[0:3],width,yerr=error[0:3],
                    color='w',ec=mysim.colors[j],
                    hatch='\\\\\\\\\\\\\\',capsize=2,label=Titles[j])
    else: plt.bar(meths+width*(j-0.5),avg,width,yerr=error,
                    color='w',ec=mysim.colors[j],
                    hatch='\\\\\\\\\\\\\\',capsize=2,label=Titles[j])
    if(i>0):plt.xticks(meths[0:3],argrithms[0:3],fontsize=8)
    else: plt.xticks(meths,argrithms,fontsize=8)
    plt.yticks(ranges[i],fontsize=8)
    plt.legend(loc=1,fontsize=8)
plt.savefig('hfs_figs/overall_res.png',bbox_inches='tight', dpi=300) 
'''