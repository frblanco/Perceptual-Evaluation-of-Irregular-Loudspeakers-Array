# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:30:23 2023

@author: FEIG
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import prince
from pandas.io.formats import style
import statsmodels.stats.multicomp as multi 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu


def normalize(arr):
    norm_arr = []
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = ((i - min(arr))/diff_arr)
        norm_arr.append(temp)
    return norm_arr

#% Data reading
# print(f"File location using __file__ variable: {os.path.realpath(os.path.dirname('ANOVA_MUSHRA.py'))}")print(f"File location using __file__ variable: {os.path.realpath(os.path.dirname('ANOVA_MUSHRA.py'))}")
# absolute path, using \ and r prefix
os.chdir(r"C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis")
# load txt files to label all values. To fit for a stats model
labels = np.loadtxt('naming.txt', dtype=str, delimiter=',', usecols=(0, 1, 2))

###read the .txt files and create a data frame - Envelopment###
# absolute path, using \ and r prefix
os.chdir(r'C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis\Envelopment')
filenames = [i for i in glob.glob("*.txt")]
participants = len(filenames)
listeners = np.arange(0, participants)
dataEnv = [np.loadtxt(file, skiprows=1, delimiter=' ', max_rows=48, usecols=(3),
                      unpack=True)
           for file in filenames]

# reshape matrix and assign columns names
dataf = pd.DataFrame(np.transpose(np.array(dataEnv)))
data_label = dataf.assign(
    Channel=labels[:, 0], Excerpt=labels[:, 1], Loudspeaker_layout=labels[:, 2])

# reshape the d dataframe suitable for statsmodels package using pd.melt
var = np.linspace(0, (len(filenames)-1), (len(filenames)))
data_label_melt = pd.melt(data_label, id_vars=[
                          'Loudspeaker_layout', 'Channel', 'Excerpt'], value_vars=var, value_name='Envelopment')

data_label_melt.rename(columns={'variable': 'Listener'}, inplace=True)

###read the .txt files and create a data frame - BAQ###
# absolute path, using \ and r prefix
os.chdir(r'C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis\BAQ')
filenames = [i for i in glob.glob("*.txt")]
dataBAQ = [np.loadtxt(file, skiprows=1, delimiter=' ', max_rows=48, usecols=(3),
                      unpack=True)
           for file in filenames]
# reshape matrix and assign columns names
dataBAQf = pd.DataFrame(np.transpose(np.array(dataBAQ)))
dataBAQ_label = dataBAQf.assign(
    Channel=labels[:, 0], Excerpt=labels[:, 1], Loudspeaker_layout=labels[:, 2])
# reshape the d dataframe suitable for statsmodels package
var = np.linspace(0, (len(filenames)-1), (len(filenames)))
dataBAQ_label_melt = pd.melt(dataBAQ_label, id_vars=[
                             'Loudspeaker_layout', 'Channel', 'Excerpt'], value_vars=var, value_name='BAQ')
dataBAQ_label_melt.rename(columns={'variable': 'Listener'}, inplace=True)

presentation_order = np.array([['BAQ,Env.',
'Env.,BAQ',
'Env.,BAQ',
'Env.,BAQ',
'BAQ,Env.',
'BAQ,Env.',
'BAQ,Env.',
'Env.,BAQ',
'Env.,BAQ',
'Env.,BAQ',
'BAQ,Env.',
'BAQ,Env.',
'Env.,BAQ',
'BAQ,Env.',
'Env.,BAQ',
'BAQ,Env.']])



plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_theme(style='ticks', palette='deep', font='Arial', font_scale=1.5, color_codes=True, rc=None)
#%%

HitReference_Env_p = (np.array([[len(data_label_melt[(data_label_melt["Loudspeaker_layout"] == "Reference") &
                                                   (data_label_melt["Envelopment"] == 100) & (data_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;
HitReference_BAQ_p = (np.array([[len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "Reference") &
                                                      (dataBAQ_label_melt["BAQ"] == 100) & (dataBAQ_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;
participants = len(listeners)
N=HitReference_BAQ_p.T.shape
listeners =np.array([np.arange(0, participants)])


a=np.array(participants*[['BAQ']])
b=np.array(participants*[['Envelopment']])
HitReference_BAQ_p=np.c_[HitReference_BAQ_p.T, a,listeners.T,presentation_order.T];
HitReference_Env_p=np.c_[HitReference_Env_p.T, b,listeners.T,presentation_order.T];
Ref_eff_dataframe = pd.DataFrame(np.r_[HitReference_BAQ_p,HitReference_Env_p])


Ref_eff_dataframe.rename(columns={0: 'Hidden Reference Identified %', 1: 'Attribute', 2: 'Listener',3: 'Presentation Order' }, inplace=True)
gx = sns.boxplot(y=Ref_eff_dataframe['Hidden Reference Identified %'].astype(float), x='Presentation Order', hue='Attribute',
                 data=Ref_eff_dataframe)
plt.ylim(0,100)
plt.show()
gx = sns.barplot(y=Ref_eff_dataframe['Hidden Reference Identified %'].astype(float), x='Listener', hue='Attribute',
                 data=Ref_eff_dataframe)
plt.ylim(0,100)
plt.show()

#%% Remove non-reliable subjects
listeners_remove = [2,3,7,8,11,12,14,15]
index_remove = len(listeners_remove)
presentation_order =np.array([np.delete(presentation_order,listeners_remove)])
listeners =np.array([np.delete(listeners,listeners_remove)])

for f in listeners_remove:
    data_label_melt=data_label_melt[data_label_melt.Listener != int(f)]
    dataBAQ_label_melt=dataBAQ_label_melt[dataBAQ_label_melt.Listener != int(f)]
    listeners =np.delete(listeners,np.where(listeners == int(f)))


#%% Statistics summary
Summary_statistics_Env= data_label_melt['Envelopment'].agg(["count", "min", "max", "median", "mean", "skew"])
Summary_statistics_BAQ= dataBAQ_label_melt['BAQ'].agg(["count", "min", "max", "median", "mean", "skew"])

###Shapiro's test (normality testq )
##first test of normality
shapiro(dataBAQ_label_melt['BAQ'])
shapiro(data_label_melt['Envelopment'])
plt.hist(data_label_melt['Envelopment'])
plt.xlabel('Envelopment')
plt.ylabel('Frequency')
plt.show()
plt.hist(dataBAQ_label_melt['BAQ'])
plt.xlabel('BAQ')
plt.ylabel('Frequency')
plt.show()


#%%second test of normality. Just took the 100 values and references-

y_Env=data_label_melt[(data_label_melt["Envelopment"] != 100) & (data_label_melt["Loudspeaker_layout"] != 'Reference') ]
y_BAQ=dataBAQ_label_melt[(dataBAQ_label_melt["BAQ"] != 100) & (dataBAQ_label_melt["Loudspeaker_layout"] != 'Reference') ]
shapiro(y_Env['Envelopment'])
shapiro(y_BAQ['BAQ'])
plt.hist(y_Env['Envelopment'])
plt.xlabel('Envelopment')
plt.ylabel('Frequency')
plt.show()
plt.hist(y_BAQ['BAQ'])
plt.xlabel('BAQ')
plt.ylabel('Frequency')
plt.show()

#shapiro((np.power(y['Envelopment'],2)))
#%%third iteration Feature_scaling

# copy the data
y_Env_norm = y_Env.copy()
y_BAQ_norm = y_BAQ.copy()
# apply normalization techniques by Column 1
column = 'Envelopment'
y_Env_norm[column] = (y_Env_norm[column] - y_Env_norm[column].min()) / (y_Env_norm[column].max() - y_Env_norm[column].min())    
column = 'BAQ'
y_BAQ_norm[column] = (y_BAQ_norm[column] - y_BAQ_norm[column].min()) / (y_BAQ_norm[column].max() - y_BAQ_norm[column].min())    


shapiro(np.array(y_Env_norm['Envelopment']))
shapiro(np.array(y_BAQ_norm['BAQ']))
plt.hist(y_Env_norm['Envelopment'])
plt.xlabel('Envelopment')
plt.ylabel('Frequency')
plt.show()
plt.hist(y_BAQ_norm['BAQ'])
plt.xlabel('BAQ')
plt.ylabel('Frequency')
plt.show()


#%%fourth iteration log

# copy the data
y_Env_norm = y_Env.copy()
y_BAQ_norm = y_BAQ.copy()
# apply normalization techniques by Column 1
column = 'Envelopment'
y_Env_norm[column] =  np.log(y_Env_norm[column])
column = 'BAQ'
y_BAQ_norm[column] = np.log(y_BAQ_norm[column])


shapiro(np.array(y_Env_norm['Envelopment']))
shapiro(np.array(y_BAQ_norm['BAQ']))
plt.hist(y_Env_norm['Envelopment'])
plt.xlabel('Envelopment')
plt.ylabel('Frequency')
plt.show()
plt.hist(y_BAQ_norm['BAQ'])
plt.xlabel('BAQ')
plt.ylabel('Frequency')
plt.show()

#%%nonpamrametric
y_Env_norm=data_label_melt

y_BAQ_norm=dataBAQ_label_melt
### Envelopment
#for channel
data_group1c = y_Env_norm[y_Env_norm['Channel']=='2chn']
data_group2c = y_Env_norm[y_Env_norm['Channel']=='5chn']
data_group3c = y_Env_norm[y_Env_norm['Channel']=='12chn']
result = stats.kruskal(data_group1c.Envelopment, data_group2c.Envelopment,data_group3c.Envelopment)
print(result)
#for loudspeaker layout 
data_group1l = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Angular Dist.']
data_group2l = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Distance Dist.']
data_group3l = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Mixed Dist.']
data_group4l = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Reference']
result = stats.kruskal(data_group1l.Envelopment, data_group2l.Envelopment,data_group3l.Envelopment,data_group4l.Envelopment)
print(result)
#for Excerpt
data_group1e = y_Env_norm[y_Env_norm['Excerpt'] == 'Poorly Corr.']
data_group2e = y_Env_norm[y_Env_norm['Excerpt'] == 'Transient']
data_group3e = y_Env_norm[y_Env_norm['Excerpt'] == 'Highly Corr.(P)']
data_group4e = y_Env_norm[y_Env_norm['Excerpt'] == 'Highly Corr.(J)']
result = stats.kruskal(data_group1e.Envelopment, data_group2e.Envelopment,data_group3e.Envelopment,data_group4e.Envelopment)
print(result)
#A:B
result = stats.kruskal(data_group1c.Envelopment, data_group2c.Envelopment,data_group3c.Envelopment,data_group1l.Envelopment, data_group2l.Envelopment,data_group3l.Envelopment,data_group4l.Envelopment)
print(result)
#A:C
result = stats.kruskal(data_group1e.Envelopment, data_group2e.Envelopment,data_group3e.Envelopment,data_group4e.Envelopment,data_group1c.Envelopment, data_group2c.Envelopment,data_group3c.Envelopment)
print(result)
#B:C
result = stats.kruskal(data_group1l.Envelopment, data_group2l.Envelopment,data_group3l.Envelopment,data_group4l.Envelopment,data_group1e.Envelopment, data_group2e.Envelopment,data_group3e.Envelopment,data_group4e.Envelopment)
print(result)
#A:B:C
result = stats.kruskal(data_group1c.Envelopment, data_group2c.Envelopment,data_group3c.Envelopment,data_group1l.Envelopment, data_group2l.Envelopment,data_group3l.Envelopment,data_group4l.Envelopment,data_group1e.Envelopment, data_group2e.Envelopment,data_group3e.Envelopment,data_group4e.Envelopment)
print(result)



### BAQ
#for channel
data_group1c = y_BAQ_norm[y_BAQ_norm['Channel']=='2chn']
data_group2c= y_BAQ_norm[y_BAQ_norm['Channel']=='5chn']
data_group3c = y_BAQ_norm[y_BAQ_norm['Channel']=='12chn']
result = stats.kruskal(data_group1c.BAQ, data_group2c.BAQ,data_group3c.BAQ)
print(result)
#for loudspeaker layout 
data_group1l = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Angular Dist.']
data_group2l = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Distance Dist.']
data_group3l = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Mixed Dist.']
data_group4l = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Reference']
result = stats.kruskal(data_group1l.BAQ, data_group2l.BAQ,data_group3l.BAQ,data_group4l.BAQ)
print(result)
#for Excerpt
data_group1e = y_BAQ_norm[y_BAQ_norm['Excerpt'] == 'Poorly Corr.']
data_group2e = y_BAQ_norm[y_BAQ_norm['Excerpt'] == 'Transient']
data_group3e = y_BAQ_norm[y_BAQ_norm['Excerpt'] == 'Highly Corr.(P)']
data_group4e = y_BAQ_norm[y_BAQ_norm['Excerpt'] == 'Highly Corr.(J)']
result = stats.kruskal(data_group1e.BAQ, data_group2e.BAQ,data_group3e.BAQ,data_group4e.BAQ)
print(result)

#A:B
result = stats.kruskal(data_group1c.BAQ, data_group2c.BAQ,data_group3c.BAQ,data_group1l.BAQ, data_group2l.BAQ,data_group3l.BAQ,data_group4l.BAQ)
print(result)
#A:C
result = stats.kruskal(data_group1e.BAQ, data_group2e.BAQ,data_group3e.BAQ,data_group4e.BAQ,data_group1c.BAQ, data_group2c.BAQ,data_group3c.BAQ)
print(result)
#B:C
result = stats.kruskal(data_group1l.BAQ, data_group2l.BAQ,data_group3l.BAQ,data_group4l.BAQ,data_group1e.BAQ, data_group2e.BAQ,data_group3e.BAQ,data_group4e.BAQ)
print(result)
#A:B:C
result = stats.kruskal(data_group1c.BAQ, data_group2c.BAQ,data_group3c.BAQ,data_group1l.BAQ, data_group2l.BAQ,data_group3l.BAQ,data_group4l.BAQ,data_group1e.BAQ, data_group2e.BAQ,data_group3e.BAQ,data_group4e.BAQ)
print(result)




#%%
###post hoc analysis 
##Env
groupA = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Angular Dist.']
groupB = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Distance Dist.']
groupC = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Mixed Dist.']
groupD = y_Env_norm[y_Env_norm['Loudspeaker_layout'] == 'Reference']
#A:B
result=mannwhitneyu(groupA.Envelopment,groupB.Envelopment,alternative='two-sided',method='exact')
print(result)
#A:C
result=mannwhitneyu(groupA.Envelopment,groupC.Envelopment,alternative='two-sided',method='exact')
print(result)
#A:D
result=mannwhitneyu(groupA.Envelopment,groupD.Envelopment,alternative='two-sided',method='exact')
print(result)
#B:C
result=mannwhitneyu(groupB.Envelopment,groupC.Envelopment,alternative='two-sided',method='exact')
print(result)
#B:D
result=mannwhitneyu(groupB.Envelopment,groupD.Envelopment,alternative='two-sided',method='exact')
print(result)
#C:D
result=mannwhitneyu(groupC.Envelopment,groupD.Envelopment,alternative='two-sided',method='exact')
print(result)
##BAQ
groupA = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Angular Dist.']
groupB = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Distance Dist.']
groupC = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Mixed Dist.']
groupD = y_BAQ_norm[y_BAQ_norm['Loudspeaker_layout'] == 'Reference']
#A:B
result=mannwhitneyu(groupA.BAQ,groupB.BAQ,alternative='two-sided',method='exact')
print(result)
#A:C
result=mannwhitneyu(groupA.BAQ,groupC.BAQ,alternative='two-sided',method='exact')
print(result)
#A:D
result=mannwhitneyu(groupA.BAQ,groupD.BAQ,alternative='two-sided',method='exact')
print(result)
#B:C
result=mannwhitneyu(groupB.BAQ,groupC.BAQ,alternative='two-sided',method='exact')
print(result)
#B:D
result=mannwhitneyu(groupB.BAQ,groupD.BAQ,alternative='two-sided',method='exact')
print(result)
#C:D
result=mannwhitneyu(groupC.BAQ,groupD.BAQ,alternative='two-sided',method='exact')
print(result)
#%% Linear mixed effect model 

mdf = smf.mixedlm("""BAQ ~ C(Listener)""", y_BAQ_norm,groups="Channel").fit()


print(mdf.summary())
plt.scatter(y_BAQ_norm['BAQ'] - mdf.resid, mdf.resid, alpha = 0.5)
plt.title("Residual vs. Fitted in Python")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig('python_plot.png',dpi=300)
plt.show()

