# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:27:59 2019

@author: qiangqiang ouyang
"""

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from statsmodels.stats.libqsturng import psturng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#df=np.load('dataset/analysis_expdata.npy')
fpath='dataset/analysis_expdata_new.txt'
headers=np.array(['stypes','meths','soft','normal','hard','acc','avg_f','peak_f','comptime','tumortime'])
df=pd.read_csv(fpath,header=None,names=headers)
df.to_csv("dataset/data.csv")
analysis1=['acc','avg_f','comptime','tumortime']
analysis2=['soft','normal','hard']

for i in range(len(analysis1)):
    dpfc=headers[headers[:]==analysis1[i]][0]
    print(" \033[1;41m" +'Anove res: '+dpfc+  "\033[0m")
    formula = dpfc+'~C(stypes)+C(meths)+C(stypes):C(meths)'
    anova_results = anova_lm(ols(formula,df).fit())
    print(anova_results)
    print(pairwise_tukeyhsd(df[dpfc], df['meths'], alpha=0.05))


for i in range(len(analysis2)):
    dpfc=headers[headers[:]==analysis2[i]][0]
    
    print(" \033[1;41m" +'Anove res: '+dpfc+  "\033[0m")
    
    
    formula = dpfc+'~C(stypes)+C(meths)+C(stypes):C(meths)'
    
    anova_results = anova_lm(ols(formula,df).fit())
    print(anova_results)
    print(pairwise_tukeyhsd(df[dpfc], df['meths'], alpha=0.05))
