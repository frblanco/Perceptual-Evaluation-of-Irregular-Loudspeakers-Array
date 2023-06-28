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
from bioinfokit.analys import stat

def normalize(arr):
    norm_arr = []
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = ((i - min(arr))/diff_arr)
        norm_arr.append(temp)
    return norm_arr

#% Data reading
os.chdir(r"C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis")
# load txt files to label all values.
labels = np.loadtxt('naming.txt', dtype=str, delimiter=',', usecols=(0, 1, 2))
errors = np.loadtxt('positional_error.txt', dtype=float, delimiter=',', usecols=(0, 1))

###read the .txt files and create a data frame###
## Envelopment file ##
#read file
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
data_label["Azimuth Error"]=errors[:, 0]
data_label["Distance Error"]=errors[:, 1]

# reshape the d dataframe suitable for statsmodels package using pd.melt
var = np.linspace(0, (len(filenames)-1), (len(filenames)))
data_label_melt = pd.melt(data_label, id_vars=[
                          'Loudspeaker_layout', 'Channel', 'Excerpt','Azimuth Error','Distance Error'], value_vars=var, value_name='Envelopment')

data_label_melt.rename(columns={'variable': 'Listener'}, inplace=True)

## BAQ file ##
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
dataBAQ_label["Azimuth Error"]=errors[:,0]
dataBAQ_label["Distance Error"]=errors[:,1]
# reshape the d dataframe suitable for statsmodels package
var = np.linspace(0, (len(filenames)-1), (len(filenames)))
dataBAQ_label_melt = pd.melt(dataBAQ_label, id_vars=[
                             'Loudspeaker_layout', 'Channel', 'Excerpt','Azimuth Error','Distance Error'], value_vars=var, value_name='BAQ')
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

#set image parameters
plt.rcParams['figure.figsize'] = [7, 4.5]
plt.rcParams['figure.dpi'] = 300
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_theme(style='ticks', palette='deep', font='Arial', font_scale=1.5, color_codes=True, rc=None)
sns.set_style("whitegrid")
#%% Analyse how many hidden references each participants identified.




HitReference_Env_p = (np.array([[len(data_label_melt[(data_label_melt["Loudspeaker_layout"] == "Reference") &
                                                   (data_label_melt["Envelopment"] == 100) & (data_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;
HitReference_BAQ_p = (np.array([[len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "Reference") &
                                                      (dataBAQ_label_melt["BAQ"] == 100) & (dataBAQ_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;

#create participants arrangement
participants = len(listeners)
N=HitReference_BAQ_p.T.shape
listeners =np.array([np.arange(0, participants)])

#create dataframe with the % of Hidden reference identified
a=np.array(participants*[['BAQ']])
b=np.array(participants*[['Envelopment']])
HitReference_BAQ_p=np.c_[HitReference_BAQ_p.T, a,listeners.T,presentation_order.T];
HitReference_Env_p=np.c_[HitReference_Env_p.T, b,listeners.T,presentation_order.T];
Ref_eff_dataframe = pd.DataFrame(np.r_[HitReference_BAQ_p,HitReference_Env_p])
Ref_eff_dataframe.rename(columns={0: 'Hidden Reference Identified %', 1: 'Attribute', 2: 'Listener',3: 'Presentation Order' }, inplace=True)

#barplot for each participant
gx = sns.barplot(y=Ref_eff_dataframe['Hidden Reference Identified %'].astype(float), x='Presentation Order', hue='Attribute',
                 data=Ref_eff_dataframe)
plt.ylim(0,100)
plt.show()
hx = sns.barplot(y=Ref_eff_dataframe['Hidden Reference Identified %'].astype(float), x='Listener', hue='Attribute',
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
#Envelopment
for i in ["2chn","5chn","12chn"]:
    for k in ["Choir","Movie","Pop","Jazz"]:
        normality = data_label_melt[(data_label_melt['Channel'] == i) & (dataBAQ_label_melt['Excerpt']== k)]
        normality['Envelopment'] = (normality['Envelopment'] - normality['Envelopment'].min()) / (normality['Envelopment'].max() - normality['Envelopment'].min())
        result = shapiro(normality['Envelopment'])
        print("For Envelopment ratings with",i,"and for excerpt",k,"the p-value is = ",result.pvalue)
#BAQ
for i in ["2chn","5chn","12chn"]:
    for k in ["Choir","Movie","Pop","Jazz"]:
        normality = dataBAQ_label_melt[(dataBAQ_label_melt['Channel'] == i) & (dataBAQ_label_melt['Excerpt']== k)]
        normality['BAQ'] = (normality['BAQ'] - normality['BAQ'].min()) / (normality['BAQ'].max() - normality['BAQ'].min())  
        result = shapiro(normality['BAQ'])
        print("For BAQ ratings with ",i," and for excerpt",k,"the p-value is = ",result.pvalue)
plt.hist(data_label_melt['Envelopment'])
plt.xlabel('Envelopment')
plt.ylabel('Frequency')
plt.show()
plt.hist(dataBAQ_label_melt['BAQ'])
plt.xlabel('BAQ')
plt.ylabel('Frequency')
plt.show()


#%% Box plots
##Envelopmentc
sns.set_theme(style="whitegrid")
ax = sns.boxplot(y='Envelopment', x="Loudspeaker_layout",
                 data=data_label_melt)
ax = sns.swarmplot(y='Envelopment', x='Loudspeaker_layout',
                   data=data_label_melt)
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
    

#%% Participants rating behaviour/ Which excerpt did my participants missidentified the reference
sns.set_style("whitegrid")

###Envelopment
plt.figure()
ex = sns.boxplot(y='Envelopment', x='Listener',notch='true', data=data_label_melt)
ex.set_title('Participant ratings')
plt.show()

plt.figure()
y_Env=data_label_melt[(data_label_melt["Loudspeaker_layout"] == 'Reference') ]
ex = sns.boxplot(y='Envelopment', x='Listener', hue='Excerpt',data=y_Env)
ex.set_title('Hidden Reference ratings')
plt.show()


###BAQ
plt.figure()
fx = sns.boxplot(y='BAQ', x='Listener',notch='true',data=dataBAQ_label_melt)
fx.set_title('Participant ratings')
plt.show()

plt.figure()
y_BAQ=dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == 'Reference') ]
fx = sns.boxplot(y='BAQ', x='Listener', hue='Excerpt',                  data=y_BAQ)
fx.set_title('Hidden Reference ratings')
plt.show()


#%% ANOVA
### per channel number
y_Env_norm=data_label_melt
y_BAQ_norm=dataBAQ_label_melt

##Env
groupA = y_Env_norm[y_Env_norm['Channel'] == '2chn']
groupB = y_Env_norm[y_Env_norm['Channel'] == '5chn']
groupC = y_Env_norm[y_Env_norm['Channel'] == '12chn']
#2chn
res = stat()
res.anova_stat(df=groupA,res_var='Envelopment',anova_model ='Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
print(res.anova_summary)
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - Envelopment for 2 channels')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
#5chn
res = stat()
res.anova_stat(df=groupB,res_var='Envelopment',anova_model ='Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
print(res.anova_summary)
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - Envelopment for 5 channels')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
#12chn
res = stat()
res.anova_stat(df=groupC,res_var='Envelopment',anova_model ='Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
print(res.anova_summary)
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - Envelopment for 12 channels')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

##BAQ
#2chn
res = stat()
res.anova_stat(df=groupA,res_var='BAQ',anova_model ='BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
print(res.anova_summary)
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - BAQ for 2 channels')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

#5chn
res = stat()
res.anova_stat(df=groupB,res_var='BAQ',anova_model ='BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
print(res.anova_summary)
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - BAQ for 5 channels')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.title('Residual histogram - Envelopment for 12 channels')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

###12chn
res = stat()
res.anova_stat(df=groupC,res_var='BAQ',anova_model ='BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
print(res.anova_summary)
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - BAQ for 12 channels')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.title('Residual histogram - BAQ for 12 channels')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()


#%%- ANOVA
##full model
###Envelopment
res = stat()
res.anova_stat(df=data_label_melt,res_var='Envelopment',anova_model ='Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)+C(Loudspeaker_layout):C(Channel)+C(Channel):C(Excerpt)+C(Channel):C(Excerpt):C(Loudspeaker_layout)')
res.anova_summary

res = stat()
res.anova_stat(df=data_label_melt,res_var='Envelopment',anova_model ='Envelopment ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
res.anova_summary
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - Envelopment')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
#interaction plot
fig=interaction_plot( x=data_label_melt['Loudspeaker_layout'], trace=data_label_melt['Channel'], response=data_label_melt['Envelopment'])
plt.ylim([0,100])


###BAQ
res = stat()
res.anova_stat(df=dataBAQ_label_melt,res_var='BAQ',anova_model ='BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)+C(Loudspeaker_layout):C(Channel)+C(Channel):C(Excerpt)+C(Channel):C(Excerpt):C(Loudspeaker_layout)')
res.anova_summary

res = stat()
res.anova_stat(df=dataBAQ_label_melt,res_var='BAQ',anova_model ='BAQ ~ C(Loudspeaker_layout) + C(Excerpt) + C(Loudspeaker_layout):C(Excerpt)')
res.anova_summary
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.title('Q-Q Plot - BAQ')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()
# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.title('Residual histogram - BAQ ')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
#interaction plot

fig=interaction_plot( x=dataBAQ_label_melt['Loudspeaker_layout'], trace=dataBAQ_label_melt['Channel'], response=dataBAQ_label_melt['BAQ'])
plt.ylim([0,100])

#%%post-hoc

tukey_Env = pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=data_label_melt['Loudspeaker_layout'], alpha=0.05)
print(tukey_Env)
tukey_Env = pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=data_label_melt['Excerpt'], alpha=0.05)
print(tukey_Env)
tukey_Env = pairwise_tukeyhsd(endog=data_label_melt['Envelopment'], groups=data_label_melt['Channel'], alpha=0.05)
print(tukey_Env)


tukey_BAQ = pairwise_tukeyhsd(endog=dataBAQ_label_melt['BAQ'], groups=dataBAQ_label_melt['Loudspeaker_layout'], alpha=0.05)
print(tukey_BAQ)
tukey_BAQ = pairwise_tukeyhsd(endog=dataBAQ_label_melt['BAQ'], groups=dataBAQ_label_melt['Excerpt'], alpha=0.05)
print(tukey_BAQ)
tukey_BAQ = pairwise_tukeyhsd(endog=dataBAQ_label_melt['BAQ'], groups=dataBAQ_label_melt['Channel'], alpha=0.05)
print(tukey_BAQ)

#%% Linear mixed effect model 

y_Env_norm=data_label_melt
y_BAQ_norm=dataBAQ_label_melt

###BAQ attribute

##Create the model
model_BAQ =smf.mixedlm("""BAQ ~ -1 + C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", y_BAQ_norm,groups="Listener").fit()
print(model_BAQ.summary())

##Reduce model to significant effects
model_BAQ =smf.mixedlm("""BAQ ~ -1 + C(Loudspeaker_layout)""", y_BAQ_norm,groups="Listener").fit()
print(model_BAQ.summary())
r=model_BAQ.fe_params

##Plot the means
plt.figure() 
plt.errorbar(["Ang. Dist.","Dist. Dist.","Mixed Dist.","Reference"],r.array,yerr=2*model_BAQ.bse_fe,fmt='o');
plt.title("Linear mixed effect model means (Listener as random effect)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("BAQ")
plt.show()
##plot residuals vs preddicted
plt.figure()
plt.scatter(model_BAQ.resid, model_BAQ.fittedvalues, alpha = 0.5)
plt.title("Residual vs. Fitted in Python")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig('python_plot.png',dpi=300)
plt.show()
fig = plt.figure(figsize = (16, 9))
##Residuals kdistribution
plt.figure()
ax = sns.distplot(model_BAQ.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")
##Q-Q PLot
fig2 = plt.figure(figsize = (16, 9))
ax = fig.add_subplot(111)
sm.qqplot(model_BAQ.resid, dist = stats.norm, line = 's', ax = ax)
ax.set_title("Q-Q Plot")



###Envelopment attribute
##Create the model
model_Env =smf.mixedlm("""Envelopment ~ -1 + C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", y_Env_norm,groups="Listener").fit()
print(model_Env.summary())
model_Env= smf.mixedlm("""Envelopment~ -1 + C(Loudspeaker_layout)""", y_Env_norm,groups="Listener").fit()
print(model_Env.summary())

##Plot the means
plt.figure() 
plt.errorbar(["Ang. Dist.","Dist. Dist.","Mixed Dist.","Reference"],r.array,yerr=2*model_BAQ.bse_fe,fmt='o');
plt.title("Linear mixed effect model means (Listener as random effect)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Envelopment")
plt.show()
##plot residuals vs preddicted
plt.figure()
plt.scatter(model_BAQ.resid, model_BAQ.fittedvalues, alpha = 0.5)
plt.title("Residual vs. Fitted in Python")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig('python_plot.png',dpi=300)
plt.show()
fig = plt.figure(figsize = (16, 9))
##Residuals kdistribution
plt.figure()
ax = sns.distplot(model_BAQ.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")
##Q-Q PLot
fig2 = plt.figure(figsize = (16, 9))
ax = fig.add_subplot(111)
sm.qqplot(model_BAQ.resid, dist = stats.norm, line = 's', ax = ax)
ax.set_title("Q-Q Plot")


#%% Correlation analysis

def correlation(x,y):
    correlation = y.corr(x)
    print(correlation)
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))
             (np.unique(x)), color='red')
    plt.show()
##BAQ and Azimuth error
correlation(dataBAQ_label_melt['Azimuth Error'],dataBAQ_label_melt.BAQ)
##BAQ and Distance error
correlation(dataBAQ_label_melt['Distance Error'],dataBAQ_label_melt.BAQ)
##Env and Azimuth error
correlation(data_label_melt['Azimuth Error'],data_label_melt.Envelopment)
##Env and Distance error
correlation(data_label_melt['Distance Error'],data_label_melt.Envelopment)


# Correlation Matrix formation
data_label_melt['BAQ'] = dataBAQ_label_melt['BAQ']
corr_matrix = data_label_melt.corr()
print(corr_matrix)
 
#Using heatmap to visualize the correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, 
            square=True, linewidths=.5,annot=True, )

plt.title('Correlation between ratings and positional errors')
plt.xticks(rotation=45, ha='right')


#%% Export data to .txt

data = y_Env
data['BAQ'] = y_BAQ['BAQ']

path =r"C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis\data.txt"
data.to_csv(path,header=True,sep=',',index=False)
# with open(path,'a') as f:
#     data_string = data.to_string(index=False)
#     f.write(data_string)
    
