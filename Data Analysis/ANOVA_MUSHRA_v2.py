# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:46:38 2023

@author: frblancoi
Three-way ANOVA for a MUSHRA test
"""
%clear

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
#%% Data reliability

### How many references were a correct hit during the test

# Envelopment
HitReference_Env = len(data_label_melt[(data_label_melt["Loudspeaker_layout"] == "Reference") &
                                       (data_label_melt["Envelopment"] == 100)])
Ref_eff_Env = HitReference_Env/(participants*12)

HitReference_Env_p = (np.array([[len(data_label_melt[(data_label_melt["Loudspeaker_layout"] == "Reference") &
                                                   (data_label_melt["Envelopment"] == 100) & (data_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;
Ref_eff_Env = HitReference_Env/(participants*12)

print(Ref_eff_Env, 'were a correct hit in the Env test by the participants')

# BAQ
HitReference_BAQ = len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "Reference") &
                                          (dataBAQ_label_melt["BAQ"] == 100)])
Ref_eff_BAQ = HitReference_BAQ/(participants*12);

HitReference_BAQ_p = (np.array([[len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "Reference") &
                                                      (dataBAQ_label_melt["BAQ"] == 100) & (dataBAQ_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;
print(Ref_eff_BAQ, 'were a correct hit in the BAQ test by the participants')
Ref_eff_Env = HitReference_BAQ/(participants*12)
# Total
Ref_eff = (HitReference_Env+HitReference_BAQ)/(participants*12*2)


###errorplots
listeners = np.array([listeners]).flatten()
Env_yerr = []
mean_Env= []
std_Env= []
BAQ_yerr = []
mean_BAQ= []
std_BAQ= []

for i in listeners:
    y=dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "Reference") & (dataBAQ_label_melt["Listener"] == int(i))]
    mean_BAQ.append(y['BAQ'].mean())
    std_BAQ.append(y['BAQ'].std())
    n=y.shape[0]
    BAQ_yerr.append(std_BAQ[-1] / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1))
    y=data_label_melt[(data_label_melt["Loudspeaker_layout"] == "Reference") & (data_label_melt["Listener"] == int(i))]
    
    mean_Env.append(y['Envelopment'].mean())
    std_Env.append(y['Envelopment'].std())
    n=y.shape[0]
    Env_yerr.append(std_Env[-1] / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1))


gx=plt.errorbar(np.arange(0, listeners.size),mean_BAQ,yerr=np.array(BAQ_yerr).flatten(),marker='s', label='BAQ')
plt.xlabel('Listener')
plt.ylabel('Hidden Reference Identified %')
gx=plt.errorbar(np.arange(0, listeners.size),mean_Env,yerr=np.array(Env_yerr).flatten(),marker='d', label='Envelopment')

plt.ylim(0,100)
plt.legend(loc='lower right')

plt.show()





### show the reference hit for different presentation order

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
   
#%% Plot the Reference identified per test page to evaluate traning effects.


##

##
participants = len(filenames)
listeners = [np.arange(0, participants)]
listeners=np.array(np.delete(listeners,listeners_remove))
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12];
y = data_label_melt[['Loudspeaker_layout', 'Envelopment', 'Listener']]

plt.figure()
count=0;
for f in listeners:
        ya=(y.loc[(y['Listener'] == f) & (y['Loudspeaker_layout'] == 'Reference')])
        ya['Excerpt presentation'] = x1;
        plt.xlabel('Test pages')
        plt.ylabel('Hidden Reference Env. Rating')
        gx=plt.plot(np.array(ya['Excerpt presentation']),np.array(ya['Envelopment']),marker='d', label='Listener ' + str(count))
        count=count+1;
plt.legend(loc="lower right")
 

y1 = dataBAQ_label_melt[['Loudspeaker_layout', 'BAQ', 'Listener']]
plt.figure()
count=0;
for f in listeners:
        yb=(y1.loc[(y['Listener'] == f) & (y1['Loudspeaker_layout'] == 'Reference')])
        yb['Excerpt presentation'] = x1;
        plt.xlabel('Test pages')
        plt.ylabel('Hidden Reference BAQ Rating')
        gx=plt.plot(np.array(yb['Excerpt presentation']),np.array(yb['BAQ']),marker='d', label='Listener ' + str(count))
        count=count+1;
plt.legend(loc="lower right")

#%% Box plots
##Envelopmentc
sns.set_theme(style="whitegrid")
ax = sns.boxplot(y='Envelopment', x="Loudspeaker_layout",
                 data=data_label_melt)
ax = sns.swarmplot(y='Envelopment', x='Loudspeaker_layout',
                   data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-Loudspeaker_layout.png')
plt.show()
bx = sns.boxplot(y='Envelopment', x='Excerpt',
                 data=data_label_melt, color='#99c2a2')
bx = sns.swarmplot(y='Envelopment', x='Excerpt',
                   data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-Excerpt.png')
plt.show()
cx = sns.boxplot(y='Envelopment', x='Channel',
                 data=data_label_melt, color='#99c2a2')
cx = sns.swarmplot(y='Envelopment', x='Channel',
                   data=data_label_melt, color='#7d0013')
plt.savefig('boxplot-Envelopment-Channel.png')
plt.show()

dx = sns.boxplot(y='Envelopment', x='Excerpt',
                 hue="Loudspeaker_layout",  data=data_label_melt)
# ax = sns.swarmplot(y='Envelopment',x='Loudspeaker_layout', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_Envelopment_Excerpt-layout.png')
plt.show()

ex = sns.boxplot(y='Envelopment', x='Excerpt',
                 hue='Channel', data=data_label_melt)
# bx = sns.swarmplot(y='Envelopment',x='Excerpt', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_Envelopment_Excerpt-Channel.png')
plt.show()


##BAQ##

# Create boxplot per independent variable
ax = sns.boxplot(y='BAQ', x='Loudspeaker_layout',
                 data=dataBAQ_label_melt, color='#99c2a2')
ax = sns.swarmplot(y='BAQ', x='Loudspeaker_layout',
                   data=dataBAQ_label_melt, color='#7d0013')
plt.savefig('boxplot-BAQ-Loudspeaker_layout.png')
plt.show()
bx = sns.boxplot(y='BAQ', x='Excerpt',
                 data=dataBAQ_label_melt, color='#99c2a2')
bx = sns.swarmplot(y='BAQ', x='Excerpt',
                   data=dataBAQ_label_melt, color='#7d0013')
plt.savefig('boxplot-BAQ-Excerpt.png')
plt.show()
cx = sns.boxplot(y='BAQ', x='Channel',
                 data=dataBAQ_label_melt, color='#99c2a2')
cx = sns.swarmplot(y='BAQ', x='Channel',
                   data=dataBAQ_label_melt, color='#7d0013')
plt.savefig('boxplot-BAQ-Channel.png')
plt.show()

dx = sns.boxplot(y='BAQ', x='Excerpt',
                 hue="Loudspeaker_layout",  data=dataBAQ_label_melt)
# ax = sns.swarmplot(y='Envelopment',x='Loudspeaker_layout', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_BAQ_Excerpt-layout.png')
plt.show()
ex = sns.boxplot(y='BAQ', x='Excerpt', hue='Channel', data=dataBAQ_label_melt)
# bx = sns.swarmplot(y='Envelopment',x='Excerpt', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_BAQ_Excerpt-Channel.png')
plt.show()
    

#%% statistical analysis
##ANOVA Envelopment##
model1 = ols("""Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", data=data_label_melt).fit()
# perform ANOVA
ANOVA_summary15 = sm.stats.anova_lm(model1, typ=2)
print(ANOVA_summary1)

#%%

###create statistical model using statsmodel library###
model1 = ols("""Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", data=data_label_melt).fit()
# perform ANOVA
ANOVA_summary1 = sm.stats.anova_lm(model1, typ=2)
print(ANOVA_summary1)

#%%


###create statistical model using statsmodel library###
model1 = ols("""Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel)""", data=data_label_melt).fit()
# perform ANOVA
ANOVA_summary1 = sm.stats.anova_lm(model1, typ=2)
print(ANOVA_summary1)

#%%

###create statistical model using statsmodel library###
model1 = ols("""Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) + C(Loudspeaker_layout):C(Channel)""", data=data_label_melt).fit()
# perform ANOVA
ANOVA_summary1 = sm.stats.anova_lm(model1, typ=2)
print(ANOVA_summary1)
#%%

data_label_melt['Channel'] = data_label_melt['Channel'].astype('category')
data_label_melt['Loudspeaker_layout'] = data_label_melt['Loudspeaker_layout'].astype('category')
data_label_melt['Excerpt'] = data_label_melt['Excerpt'].astype('category')
data_label_melt['Listener'] = data_label_melt['Listener'].astype('category')

lm = ols("Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) + C(Loudspeaker_layout):C(Channel)",
       data=data_label_melt).fit()
lmsummary1=lm.summary()
print(lmsummary1)

#%%

# create statistical model using statsmodel library
model2 = ols("""BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", data=dataBAQ_label_melt).fit()

# perform ANOVA
ANOVA_summary2 = sm.stats.anova_lm(model2, typ=2)
print(ANOVA_summary2)

#%%

# create statistical model using statsmodel library
model2 = ols("""BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", data=dataBAQ_label_melt).fit()

# perform ANOVA
ANOVA_summary2 = sm.stats.anova_lm(model2, typ=2)
print(ANOVA_summary2)

#%%

# create statistical model using statsmodel library
model2 = ols("""BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel)""", data=dataBAQ_label_melt).fit()

# perform ANOVA
ANOVA_summary2 = sm.stats.anova_lm(model2, typ=2)
print(ANOVA_summary2)


#%%
# create statistical model using statsmodel library
model2 = ols("""BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Channel) + C(Loudspeaker_layout):C(Channel)""", data=dataBAQ_label_melt).fit()

# perform ANOVA
ANOVA_summary2 = sm.stats.anova_lm(model2, typ=2)
print(ANOVA_summary2)


#%%

dataBAQ_label_melt['Channel'] = data_label_melt['Channel'].astype('category')
dataBAQ_label_melt['Loudspeaker_layout'] = data_label_melt['Loudspeaker_layout'].astype('category')
dataBAQ_label_melt['Excerpt'] = data_label_melt['Excerpt'].astype('category')
dataBAQ_label_melt['Listener'] = data_label_melt['Listener'].astype('category')

lm2 = ols("Envelopment ~C(Loudspeaker_layout) + C(Excerpt) + C(Channel) + C(Loudspeaker_layout):C(Channel)",
       data=data_label_melt).fit()
lmsummary2=lm2.summary()
print(lmsummary2)

#%%###ANOVA plots 


#%%
sns.set_theme(style="whitegrid")
#sns.pointplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Channel',errorbar='ci,95')
sns.catplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Excerpt',col='Channel',capsize=.2, palette="YlGnBu_d", errorbar="ci,95",
    kind="point", height=6, aspect=.75,)

#%%
sns.set_theme(style="whitegrid")
#sns.pointplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Channel',errorbar='ci,95')
gx=sns.catplot(data=dataBAQ_label_melt,x='Loudspeaker_layout',y='BAQ',hue='Listener',col='Channel',capsize=.2, palette="YlGnBu_d", errorbar="ci,95",
    kind="point", height=6, aspect=.75,).despine(left=True)
#%%
fx= sns.pointplot(data=dataBAQ_label_melt,x='Loudspeaker_layout',y='BAQ',hue='Excerpt',capsize=.2, palette="YlGnBu_d", errorbar="ci,95", height=6, aspect=.75,)
#var_att= np.array(famd.column_contributions_())*100
#%%
gx= sns.pointplot(data=dataBAQ_label_melt,x='Loudspeaker_layout',y='BAQ',hue='Channel',capsize=.2, palette="YlGnBu_d", errorbar="ci,95", height=6, aspect=.75,)
#%%
hx= sns.pointplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Excerpt',capsize=.2, palette="YlGnBu_d", errorbar="ci,95", height=6, aspect=.75,)
#%%
hx= sns.pointplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Channel',capsize=.2, palette="YlGnBu_d", errorbar="ci,95", height=6, aspect=.75,)


#%% Post-hoc Analysis
Summary_statistics_Env= data_label_melt['Envelopment'].agg(["count", "min", "max", "median", "mean", "skew"])
Summary_statistics_BAQ= dataBAQ_label_melt['BAQ'].agg(["count", "min", "max", "median", "mean", "skew"])

###Shapiro's test

shapiro(dataBAQ_label_melt['BAQ'])
shapiro(data_label_melt['Envelopment'])
y=data_label_melt[data_label_melt["Envelopment"] != 100]
shapiro(y['Envelopment'])
shapiro((np.power(y['Envelopment'],2)))


y=normalize(np.array(data_label_melt['Envelopment']),np.min(np.array(data_label_melt['Envelopment'])),np.max(np.array(data_label_melt['Envelopment'])))
y=data_label_melt[(data_label_melt["Envelopment"] != 100) & (data_label_melt["Loudspeaker_layout"] != 'Reference') ]

###Mann-Whitney U test


###
Tukey_channel = pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=data_label_melt['Channel'])
print(Tukey_channel)

Tukey_Loudspeaker_layout= pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=data_label_melt['Loudspeaker_layout'])
print(Tukey_Loudspeaker_layout)

Tukey_Excerpt= pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=data_label_melt['Excerpt'])
print(Tukey_Excerpt)
# Tukey_Excerpt= pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=(data_label_melt['Loudspeaker_layout'],data_label_melt['Channel']))
# print(Tukey_channel)