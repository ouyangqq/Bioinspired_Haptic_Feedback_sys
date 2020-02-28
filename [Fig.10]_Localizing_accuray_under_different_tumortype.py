# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:30:20 2019

@author: qiangqiang ouyang
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import actuators_sensors as sa
import Receptors as rslib
import simset as mysim


def plot_sig(xstart,xend,ystart,yend,sig,colors):
    for i in range(len(xstart)):
        x = np.ones((2))*xstart[i]
        y = np.arange(ystart[i],yend[i],(yend[i]-ystart[i])*19/20)
        plt.plot(x,y,color="black",linewidth=0.5)

        x = np.arange(xstart[i],xend[i]+0.1,xend[i]-xstart[i])
        y = yend[i]+0*x
        plt.plot(x,y,color="black",linewidth=0.5)

        x0 = (xstart[i]+xend[i])/2
        y0=ystart[i]
        plt.annotate(r'%s'%sig[i], xy=(x0, y0), xycoords='data', xytext=(-1, 0),
                     textcoords='offset points', fontsize=16,color=colors[i])
        x = np.ones((2))*xend[i]
        y = np.arange(ystart[i],yend[i],(yend[i]-ystart[i])*19/20)
        plt.plot(x,y,color="black",linewidth=0.5)
        #plt.ylim(0,math.ceil(max(yend)+4))             #使用plt.ylim设置y坐标轴范围
    #     plt.xlim(math.floor(xstart)-1,math.ceil(xend)+1)
        #plt.xlabel("随便画画")         #用plt.xlabel设置x坐标轴名称
        '''设置图例位置'''
        #plt.grid(True)
    plt.show()

df=np.loadtxt('dataset/acquiring_expdata.txt',delimiter=',')

'------------- Analysis Results--------------------------'
ylabels=['Recognition rate','Average force [N]','Completion time [s]','Contacting tumor time [s]']
titles=['Soft tumor','Normal tumor','Hard tumor']
argrithms=['BHF','LHF','PDHF','NHF']
Titles=['Novices','Surgeons'] 
plt.figure(figsize=(9,3))
plt.subplots_adjust(wspace=0.3) 
plt.subplots_adjust(hspace=0.4) 
meths=np.array([1,2,3,4])
ranges=[[0,0.2,0.4,0.6,0.8,1.0],
        [0,0.2,0.4,0.6,0.8,1.0],
        [0,0.2,0.4,0.6,0.8,1.0]]
#width=0.3

for i in range(3):
    ax =plt.subplot(1,3,1+i)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.title(titles[i],fontsize=12)  # Pressing force
    plt.xlabel('Algorithm')
    if(i==0):plt.ylabel(ylabels[0])
    dbf=[]
    for m in range(4):
        tmp=df[df[:,1]==m+1,i+2]
        tmp=tmp
        #if(i==0): dbf.append(np.array([np.mean(tmp),np.var(tmp)]))
        dbf.append(np.array([np.mean(tmp),np.std(tmp)]))
    avg=np.array(dbf)[:,0]
    error=np.array(dbf)[:,1]
    #if(i==0):plt.bar(argrithms[0:3],databf[0:3,i],width,yerr=error[0:3],capsize=2)
    #else:
    plt.errorbar(meths,avg,yerr=error,marker='d',fmt='.b',elinewidth=0.5,capsize=4)
    #plt.bar(meths,avg,width,yerr=error,color='w',ec='k',hatch='\\\\\\\\\\\\\\',capsize=2)

    
    if(i==0):plot_sig([1,1,1,2],[2,3,4,3],np.array([1.15,1.25,1.35,1.05])+0.1,np.array([1.2,1.30,1.4,1.1])+0.1,['*','**','**','*'],['k','k','k','k'])
    if(i==1):plot_sig([1],[4],[1.25],[1.3],['**'],['k'])
    #if(i==2):plot_sig([1,1],[2,3],[1,1.1],[1.05,1.15],['*','*'])
    #if(i==2):plot_sig([1,1,1],[2,3,4],[3,3.3,3.6],[3.1,3.4,3.7],['*','*','*'],['r','r','r'])
    plt.yticks(ranges[i],fontsize=8)
    ax.set_xticks([0.5,4.5])
    plt.xticks(meths,argrithms,fontsize=8)
    #if(i==0):plt.legend(loc=1,fontsize=8,edgecolor='w')
plt.savefig('hfs_figs/[Fig.10]_Localizing_accuray_under_different_tumortype.png',bbox_inches='tight', dpi=300) 


