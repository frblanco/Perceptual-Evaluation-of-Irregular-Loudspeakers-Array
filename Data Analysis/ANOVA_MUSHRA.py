# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:46:38 2023

@author: frblancoi
Three-way ANOVA for a MUSHRA test
"""
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


os.chdir(r"C:\Users\frbla\OneDrive\Development\Engineering Acoustics\Special-Project\Data Analysis") # absolute path, using \ and r prefix
labels = np.loadtxt('naming.txt',dtype=str,delimiter=',', usecols =(0,1,2)) #load txt files to label all values. To fit for a stats model
#pfnames=np.loadtxt('pfnaming.txt',dtype=str,delimiter=',', usecols =(0)) #Naming 

os.chdir(r'C:\Users\frbla\OneDrive\Development\Engineering Acoustics\Special-Project\Data Analysis\Envelopment') # absolute path, using \ and r prefix
filenames = [i for i in glob.glob("*.txt")]
dataEnv = [np.loadtxt(file, skiprows=1,delimiter=' ',max_rows=48,usecols =(3), 
 unpack = True)
  for file in filenames]
dataf = pd.DataFrame(np.transpose(np.array(dataEnv)));
data_label=dataf.assign(channel=labels[:,0], excerpt=labels[:,1],Loudspeaker_layout=labels[:,2])
#     # reshape the d dataframe suitable for statsmodels package 
var = np.linspace(0,(len(filenames)-1),(len(filenames)))
data_label_melt = pd.melt(data_label, id_vars=['Loudspeaker_layout','channel','excerpt'], value_vars=var, value_name='Envelopment')


model = ols("""Envelopment ~ C(Loudspeaker_layout) + C(excerpt) + C(channel) +
               C(Loudspeaker_layout):C(excerpt) + C(Loudspeaker_layout):C(channel) + C(excerpt):C(channel) +
               C(Loudspeaker_layout):C(excerpt):C(channel)""", data=data_label_melt).fit()

ANOVA_summary1=sm.stats.anova_lm(model, typ=2)
print(ANOVA_summary1)

ax = sns.boxplot(y='Envelopment',x='Loudspeaker_layout', data=data_label_melt,color='#99c2a2')
ax = sns.swarmplot(y='Envelopment',x='Loudspeaker_layout', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-Loudspeaker_layout.png')
plt.show()
bx = sns.boxplot(y='Envelopment',x='excerpt', data=data_label_melt,color='#99c2a2')
bx = sns.swarmplot(y='Envelopment',x='excerpt', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-excerpt.png')
plt.show()
cx = sns.boxplot(y='Envelopment',x='channel', data=data_label_melt,color='#99c2a2')
cx = sns.swarmplot(y='Envelopment',x='channel', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-channel.png')
plt.show()

    
  

os.chdir(r'C:\Users\frbla\OneDrive\Development\Engineering Acoustics\Special-Project\Data Analysis\BAQ') # absolute path, using \ and r prefix
filenames = [i for i in glob.glob("*.txt")]
dataBAQ = [np.loadtxt(file, skiprows=1,delimiter=' ',max_rows=48,usecols =(3), 
 unpack = True)
  for file in filenames]
dataBAQf = pd.DataFrame(np.transpose(np.array(dataBAQ)));
dataBAQ_label=dataBAQf.assign(channel=labels[:,0], excerpt=labels[:,1],Loudspeaker_layout=labels[:,2])
#     # reshape the d dataframe suitable for statsmodels package 
var = np.linspace(0,(len(filenames)-1),(len(filenames)))
dataBAQ_label_melt = pd.melt(dataBAQ_label, id_vars=['Loudspeaker_layout','channel','excerpt'], value_vars=var, value_name='BAQ')


model = ols("""BAQ ~ C(Loudspeaker_layout) + C(excerpt) + C(channel) +
               C(Loudspeaker_layout):C(excerpt) + C(Loudspeaker_layout):C(channel) + C(excerpt):C(channel) +
               C(Loudspeaker_layout):C(excerpt):C(channel)""", data=dataBAQ_label_melt).fit()

ANOVA_summary2=sm.stats.anova_lm(model, typ=2)
print(ANOVA_summary2)

ax = sns.boxplot(y='BAQ',x='Loudspeaker_layout', data=data_label_melt,color='#99c2a2')
ax = sns.swarmplot(y='BAQ',x='Loudspeaker_layout', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-Loudspeaker_layout.png')
plt.show()
bx = sns.boxplot(y='BAQ',x='excerpt', data=data_label_melt,color='#99c2a2')
bx = sns.swarmplot(y='BAQ',x='excerpt', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-excerpt.png')
plt.show()
cx = sns.boxplot(y='BAQ',x='channel', data=data_label_melt,color='#99c2a2')
cx = sns.swarmplot(y='BAQ',x='channel', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-channel.png')
plt.show()
