import clr#clr是公共运行时环境，这个模块是与C#交互的核心
import sys
import ctypes
import time as ct
from threading import Timer  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import msvcrt
import actuators_sensors as sa

sys.path.append(r"D:\backup_research\My_workshop\Bio_inspired_haptic_controlling\HFS\HapticTools\Bioinspired_HF\bin\Debug")
clr.AddReference(r'HapticsManager')

import HapticsManager as hm   #import the name of space
instance =hm.BiosH_CS() #BiosH_CS is a class in the HapticsManager.exe
#instance.Show()
b=instance.control_actuators([0,0,0,0,0],[0,0,0,0,0]) #control actuators
print(b)

'''
fig=plt.figure(figsize=(8,5))
ax= fig.add_subplot(111)
curve,=ax.plot(time_space,sensor_data_buf,'k-',linewidth=1.5)
plt.yticks([0,500,1000])

cm = plt.get_cmap("coolwarm")
cNorm = cmx.colors.Normalize(vmin=-1, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
scalarMap.set_array([-1,1])

def Timing_plot():  
    print("start" )
    global sensor_data_buf
    #curve.set_xdata(time_space)
    curve.set_ydata(sensor_data_buf)
    #plt.plot(time_space,sensor_data_buf,'k-',linewidth=1.5)
    timer = Timer(0.2,Timing_plot)
    timer.start()
'''

T=10*60
num=int(T/0.07)
row=0
dt=0.001
time_space=np.arange(0,1,0.001)
sensor_data_buf=np.zeros(len(time_space))

count=0
rec_buf=[]

x0=sa.px0


def Timing_read():
    global count
    #global sensor_data_buf
    res=instance.read_sensor_packet() #read data from sensors
    if res!=None:
        effect=res.split('Y')[1][0:3]
        #sensor_data_buf[count]=float(res.split(',')[3][0:3])
        valule=sa.calibrated(float(effect),x0)
        print(round(ct.time()-start_time,4),valule)
        rec_buf.append(np.array([ct.time()-start_time,valule]))      
        count=count+1  
    timer = Timer(0.0001,Timing_read)
    timer.start()
    if(count>=num):
        print('Record completed')
        timer.cancel()
        np.save('rec_sensor_buf1.npy',rec_buf)

instance.Start_sensor()  
print('Start now')
start_time=ct.time()
Timing_read()        
#Timing_plot()
'''
for time in range( int(T/dt)):  #时间 T/dt 
    if (time%(int(0.1/dt))==0):
        curve.set_ydata(sensor_data_buf)
'''
