# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:49:51 2019
@author: qiangqiang ouyang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


buf=np.load('data/rec_sensor_buf1.npy')

plt.figure(figsize=(8,5))
plt.subplot(121)
plt.plot(buf[:,0],buf[:,1])
plt.xlabel('Time (s)')

'''
import clr 
clr.CompileModules("C:\pyrad\pyrad.dll", "C:\pyrad\setup.py")
'''