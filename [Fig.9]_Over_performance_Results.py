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

'-------------Overall Analysis Results--------------------------'
ylabels=['Localizing accuracy','Mean force [N]','Completion time [s]','Contacting tumor time [s]']
titles=['(a)','(b)','(c)','(d)']
argrithms=['BHF','LHF','PDHF','NHF']
Titles=['Novices','Surgeons'] 
plt.figure(figsize=(8,6))
plt.subplots_adjust(wspace=0.3) 
plt.subplots_adjust(hspace=0.4) 
meths=np.array([1,2,3,4])
ranges=[[0,0.2,0.4,0.6,0.8,1.0],
        [0,1,2,3,4,],
        [0,30,60,90,120,150,180,210,240],
        [0,1,2,3,4]]
width=0.2
sels=[5,6,8,9]

#F_k=2.8
for i in range(4):
    ax =plt.subplot(2,2,1+i)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.title(titles[i],fontsize=12)  # Pressing force
    plt.xlabel('Algorithm')
    plt.ylabel(ylabels[i])
    for j in range(2):
        dbf=[]
        accbf=[]
        for m in range(4):
            tmp=df[(df[:,0]==j)&(df[:,1]==m+1),sels[i]]
            #if (i==1):tmp=tmp/F_k
            if(i==0):accbf.append(tmp)
            #if(i==0): dbf.append(np.array([np.mean(tmp),np.var(tmp)]))
            dbf.append(np.array([np.mean(tmp),np.std(tmp)]))
        avg=np.array(dbf)[:,0]
        error=np.array(dbf)[:,1]
        #if(i==0):plt.bar(argrithms[0:3],databf[0:3,i],width,yerr=error[0:3],capsize=2)
        if(i==5):
            #plt.bar(meths+width*(j-0.5),avg,width,
            #            color='w',ec=mysim.colors[j],
            #            hatch='\\\\\\\\\\\\\\',capsize=2,label=Titles[j])
            tmp=df[(df[:,0]==j),:][1,sels[i]]
            plt.boxplot(accbf,positions=meths+width*(j-0.5),widths=0.15,labels=argrithms,whis=1.5)
            #plt.boxplot(meths+width*(j-0.5),avg)
        else:
            plt.bar(meths+width*(j-0.5),avg,width,yerr=error,
                        color='w',ec=mysim.colors[j],
                        hatch='\\\\\\\\\\\\\\',capsize=2,label=Titles[j])
    plt.xticks(meths,argrithms,fontsize=8)
    #if(i==0):plot_sig([1,1,1],[2,3,4],[1.1,1.25,1.4],[1.15,1.30,1.45],['*','**','**'],['r','r','r'])
    if(i==1):plot_sig([1,1,1],[2,3,4],np.array([8.5,9.5,10.5])/F_k,np.array([9,10,11])/F_k,['*','*','*'],['r','r','r'])
    #if(i==2):plot_sig([1,1],[2,3],[1,1.1],[1.05,1.15],['*','*'])
    if(i==3):plot_sig([1,1,1],[2,3,4],[3,3.3,3.6],[3.1,3.4,3.7],['*','*','*'],['r','r','r'])
    plt.yticks(ranges[i],fontsize=8)
    if(i==2):plt.legend(loc=2,fontsize=8,edgecolor='w')
plt.savefig('hfs_figs/overall_res.png',bbox_inches='tight', dpi=300) 


