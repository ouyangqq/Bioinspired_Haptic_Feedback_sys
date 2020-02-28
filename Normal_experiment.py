import sys
#sys.path.append(r"../tactile_receptors")
#sys.path.append(r"../Skin_mechanics")
import Receptors as receptorlib
import clr#clr是公共运行时环境，这个模块是与C#交互的核心
import ctypes
import time as ct
from threading import Timer  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from scipy.interpolate import interp1d
from PIL import Image
sys.path.append(r"D:\backup_research\My_workshop\Bio_inspired_haptic_controlling\HFS\HapticTools\Bioinspired_HF\bin\Debug")
clr.AddReference(r'HapticsManager')
import HapticsManager as hm  #load namespace from DLL
import msvcrt
#import pythoncom
#import pyHook
import random
import actuators_sensors as sa
csborad=hm.BiosH_CS()      # control and sensor board function lib loaded from the DLL file

def all_3_combinations(buf):
    # 列出所有3种的组合
    res=[]
    m=len(buf)
    for i in range(0, m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                res.append(buf[k]+buf[j]+buf[i])
    return res

count1=0
def Timing_read():
    global count1
    res=csborad.read_sensor_packet() #read data from sensors
    if ((res!=None)&(res[0]=='Y')):
        effect=res.split('Y')[1][0:3]
        if(effect.isdigit()):
            pressure=sa.presure_from_AD(float(effect),sa.px0)
            sensor_rec_buf.append(np.array([ct.time()-start_time,pressure]))  
            sampled_buf[0:sampled_num-1,:]=sampled_buf[1:,:] # left shift a bit 
            sampled_buf[-1,:]=sensor_rec_buf[-1]
    count1=count1+1 
    read_timer = Timer(0.01,Timing_read)
    if(enable_timer==True):read_timer.start()

csborad.Close_sensor()  
print('We wiil give you a normal experiment, please input the name of subject') 
Name=input()     # Input the name of subject


sampled_num=int(sa.sampled_T/sa.dfT)
sampled_buf=np.zeros([sampled_num,2])

Timeout_T=100*60
Timeout_num=int(Timeout_T/sa.dfT)
sensor_rec_buf=[]
ctrl_bf=[]


fbms=['Costom','NF','Linear','Bio']



ranm=random.sample(range(0,4),4)  # random methods

argrithms=[]

for m in range(len(ranm)):
    argrithms.append(fbms[ranm[m]]+'_training')
    argrithms.append(fbms[ranm[m]]+'_exp')


argrithms.append('FT_test')

if (Name.split('_')[1]=='test'):
    argrithms=[]
    if (Name.split('_')[2]=='C'):
        argrithms.append(fbms[0]+'_training')
    elif (Name.split('_')[2]=='N'):
        argrithms.append(fbms[1]+'_training')
    elif (Name.split('_')[2]=='L'):
        argrithms.append(fbms[2]+'_training')
    elif (Name.split('_')[2]=='B'):
        argrithms.append(fbms[3]+'_training')
    else: print('Please restart the program')






#tumor_types=['soft','normal','hard']

Locatons=all_3_combinations(['1','2','3','4','5','6','7'])
#Locatons=['123','456','789','147','258','369','238','135']
#Locatons=['1','2','3','4','5','6','7','8']
trial_num=len(argrithms) #4 trianings 4 tests and 1 finger test
#*len(tumor_types)#*len(Locatons)

ranl=random.sample(range(0,len(Locatons)),trial_num)  # random locations


trial_no=0
pds=np.zeros(sa.tsensors[0].t.size)# detected pressure data buf
count2=0
ctrl=[0,0]
enable_timer=False
start_time=0
     
def restart_data_acquiring():
    global enable_timer
    global start_time
    global sensor_rec_buf
    global ctrl_bf
    csborad.Start_sensor() 
    ct.sleep(0.5)
    start_time=ct.time()
    enable_timer=True
    sampled_buf[:]=np.zeros([sampled_num,2])
    sensor_rec_buf=[]
    ctrl_bf=[] 
    Timing_read()
    while(True):
        if(count1>sampled_num+2):break

while(trial_no<trial_num):
    argrithm=argrithms[trial_no]#combined_buf[trial_no].split('_')[0]   
    if (argrithm.split('_')[1]=='training')|(argrithm.split('_')[1]=='test'):
        set_location=str(Locatons[ranl[trial_no]])
        print('Now we will start the '+argrithm.split('_')[1]+' for '+argrithm.split('_')[0]+' , please insert the tumors as HNS:'+set_location+' for normal experiment')
        while(1):
            print('Do want to start the '+argrithm.split('_')[1]+'  now? if yes please input Y or y')
            res=input()
            if((res=='y')|(res=='Y')): break
    if (argrithm.split('_')[1]=='exp'):
        print('Now we will start the '+argrithm.split('_')[0]+' experiment.'
              '\n Please wait the experimenter to prepare tumor ')
        while(1):
            print('Has the tumor  been set according to: '+set_location+'? if yes please input Y or y')
            res=input()
            if((res=='y')|(res=='Y')): break
        while(1):
            print("Do want to start the experiennt now? if yes please input Y or y")
            res=input()
            if((res=='y')|(res=='Y')): break
    #if(trial_no)restart_data_acquiring()
    restart_data_acquiring()
    while(True):
        'Loading detected data'
        tp=(sampled_buf[-1,0]-sampled_buf[0,0])/sa.multiplier
        if (tp>sa.simT):
            f = interp1d(sampled_buf[:,0]/sa.multiplier-sampled_buf[0,0]/sa.multiplier, sampled_buf[:,1])
            pds[:]=f(sa.tsensors[0].t)#
        #res=receptorlib.butterworth_filter(1,buf,250,'low',10000)
        if(argrithm.split('_')[0]=='FT'):
            ctrl[:]=[0,0]
            ctrl_bf.append(np.hstack([ct.time()-start_time,np.round(ctrl,2)]))
            csborad.control_actuators([0,ctrl[1],0,0,ctrl[0]],[0,0,0,0,0])
            ct.sleep(0.10)
        elif(argrithm.split('_')[0]=='NF'):
            ctrl[:]=[0,0]
            ctrl_bf.append(np.hstack([ct.time()-start_time,np.round(ctrl,2)]))
            csborad.control_actuators([0,ctrl[1],0,0,ctrl[0]],[0,0,0,0,0])
            ct.sleep(0.10)
        elif(argrithm.split('_')[0]=='Costom'):
            ctrl[:]=sa.current_controlling(pds)
            ctrl_bf.append(np.hstack([ct.time()-start_time,np.round(ctrl,2)]))
            csborad.control_actuators([0,ctrl[1],0,0,ctrl[0]],[0,0,0,0,0])
            ct.sleep(0.1)
        elif(argrithm.split('_')[0]=='Linear'):
            ctrl[:]=sa.linear_controlling(pds)
            ctrl_bf.append(np.hstack([ct.time()-start_time,np.round(ctrl,2)]))
            csborad.control_actuators([0,ctrl[1],0,0,ctrl[0]],[0,0,0,0,0])
            ct.sleep(0.1)
        elif(argrithm.split('_')[0]=='Bio'):
            ctrl[:]=sa.bio_inspired_controlling(pds)
            ctrl_bf.append(np.hstack([ct.time()-start_time,np.round(ctrl,2)]))
            csborad.control_actuators([0,ctrl[1],0,0,ctrl[0]],[0,0,0,0,0])
            ct.sleep(0.06)
        'Contorl the actuators'
         #Control actuators
        count2=count2+1
        if(count2%5==0):print(ctrl_bf[-1],'_sensor:',sampled_buf[-1][1:3]) 
        'scaning keyboard '
        if msvcrt.kbhit(): 
            key=ord(msvcrt.getch())
            #End the experiment when clicking ESC 
            if(key==27):
                csborad.Close_sensor()  
                csborad.control_actuators([0,0,0,0,0],[0,0,0,0,0])
                enable_timer=False
                count1=0
                report='FF'
                while(1):
                    if argrithm.split('_')[1]=='training':
                        print("Do you think the feedback system is easy for you to discriminate the tumor and normal tissue, please input Yes or No")
                        report=input() 
                        if((report=='Yes')|(report=='No')): break
                    if (argrithm.split('_')[1]=='exp')|(argrithm.split('_')[1]=='test'):
                        print("Please input the subjest's reported tumor locations that the subject felt as format:2119-2223-3352")
                        report=input()
                        if(len(report)>=14):
                            if((report[4]=='-')&(report[9]=='-')&(report[0]>='0')&(report[0]<='9')): 
                                break
                np.save('Recorded_data/'+Name+'_'+argrithm+'_'+set_location+'_'+report+'_.npy',[sensor_rec_buf,ctrl_bf])
                print('Record completed')
                break;
            # Pressing 'CTRL+q'
            elif(key==17):  
                print('Stop for error, we will restart this trial')
                ct.sleep(2)
                enable_timer=False
                count1=0
                csborad=hm.BiosH_CS()  
                csborad.Start_sensor() 
                trial_no=trial_no-1
                break
        'Time_out'
        if(count1>=Timeout_num):
            enable_timer=False
            count1=0
            #simulation_res=tmp1
            np.save('Recorded_data/'+Name+'_'+argrithm+'_'+set_location+'_'+'timeout-recordeddata.npy',[sensor_rec_buf,ctrl_bf])
            print('Time_out and we will repeat this trial')
            trial_no=trial_no-1
            break
    print('You have completed '+str(trial_no+1)+'th trial, please wait a few seconds')
    trial_no=trial_no+1
    
   
csborad.Close_sensor() 
csborad.control_actuators([0,0,0,0,0],[0,0,0,0,0])
enable_timer=False
print('You have complete all the trials. Thank you so much\
      \n ----designed by Ouyang from SEU, BRI and CASIT')