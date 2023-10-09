# -*- coding: utf-8 -*-
"""
updated on Wed Aug 30th 15:22:23 2023

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
import pingouin as pg
import statsmodels as sms

def normalize(arr):
    norm_arr = []
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = ((i - min(arr))/diff_arr)
        norm_arr.append(temp)
    return norm_arr

#% Data reading
os.chdir(r"C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Projects\Perceptual-Evaluation-of-Irregular-Loudspeakers-Array-v2\data-analysis")
# load txt files to label all values.
labels = np.loadtxt('naming.txt', dtype=str, delimiter=',', usecols=(0, 1, 2))
errors = np.loadtxt('positional_error.txt', dtype=float, delimiter=',', usecols=(0, 1))

###read the .txt files and create a data frame###
## Envelopment file ##
#read file
os.chdir(r'C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Projects\Perceptual-Evaluation-of-Irregular-Loudspeakers-Array-v2\data-analysis\Envelopment')
filenames = [i for i in glob.glob("*.txt")]
participants = len(filenames)
listeners = np.arange(0, participants)
dataEnv = [np.loadtxt(file, skiprows=1, delimiter=' ', max_rows=48, usecols=(3),
                      unpack=True)
           for file in filenames]

# reshape matrix and assign columns names
dataf = pd.DataFrame(np.transpose(np.array(dataEnv)))
data_label = dataf.assign(
    Channel=labels[:, 0], Excerpt=labels[:, 1], Loudspeaker_layout=labels[:, 2], Attribute='Envelopment')
data_label["Azimuth Error"]=errors[:, 0]
data_label["Distance Error"]=errors[:, 1]

# reshape the d dataframe suitable for statsmodels package using pd.melt
var = np.linspace(0, (len(filenames)-1), (len(filenames)))
data_label_melt = pd.melt(data_label, id_vars=[
                          'Loudspeaker_layout', 'Channel', 'Excerpt','Azimuth Error','Distance Error','Attribute'], value_vars=var, value_name='Rating')

data_label_melt.rename(columns={'variable': 'Listener'}, inplace=True)

## BAQ file ##
# absolute path, using \ and r prefix
os.chdir(r'C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Projects\Perceptual-Evaluation-of-Irregular-Loudspeakers-Array-v2\data-analysis\BAQ')
filenames = [i for i in glob.glob("*.txt")]
dataBAQ = [np.loadtxt(file, skiprows=1, delimiter=' ', max_rows=48, usecols=(3),
                      unpack=True)
           for file in filenames]
# reshape matrix and assign columns names
dataBAQf = pd.DataFrame(np.transpose(np.array(dataBAQ)))
dataBAQ_label = dataBAQf.assign(
    Channel=labels[:, 0], Excerpt=labels[:, 1], Loudspeaker_layout=labels[:, 2],Attribute='BAQ')
dataBAQ_label["Azimuth Error"]=errors[:,0]
dataBAQ_label["Distance Error"]=errors[:,1]
# reshape the d dataframe suitable for statsmodels package
var = np.linspace(0, (len(filenames)-1), (len(filenames)))
dataBAQ_label_melt = pd.melt(dataBAQ_label, id_vars=[
                             'Loudspeaker_layout', 'Channel', 'Excerpt','Azimuth Error','Distance Error','Attribute'], value_vars=var, value_name='Rating')
dataBAQ_label_melt.rename(columns={'variable': 'Listener'}, inplace=True)

data_final=dataBAQ_label_melt.append(data_label_melt)


presentation_order = np.array([['Env.,BAQ',
'BAQ,Env.',
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




HitReference_Env_p = (np.array([[len(data_label_melt[(data_label_melt["Loudspeaker_layout"] == "No Disp.") &
                                                   (data_label_melt["Rating"] == 100) & (data_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;
HitReference_BAQ_p = (np.array([[len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "No Disp.") &
                                                      (dataBAQ_label_melt["Rating"] == 100) & (dataBAQ_label_melt["Listener"] == listener)])
                               for listener in listeners]])/12)*100;

HitReference_Env_choir = np.mean((np.array([[len(data_label_melt[(data_label_melt["Loudspeaker_layout"] == "No Disp.") &
                                                   (data_label_melt["Rating"] == 100) & (data_label_melt["Listener"] == listener) & (data_label_melt["Excerpt"] == "Choir")])
                               for listener in listeners]])/3));
HitReference_BAQ_choir = np.mean((np.array([[len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == "No Disp.") &
                                                      (dataBAQ_label_melt["Rating"] == 100) & (dataBAQ_label_melt["Listener"] == listener) & (dataBAQ_label_melt["Excerpt"] == "Choir")])
                               for listener in listeners]])/3));


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
Mean_hidden_reference = pd.to_numeric(Ref_eff_dataframe['Hidden Reference Identified %'])
#barplot for each participant
gx = sns.barplot(y=Ref_eff_dataframe['Hidden Reference Identified %'].astype(float), x='Presentation Order', hue='Attribute',
                 data=Ref_eff_dataframe)
gx.legend_.remove()
plt.ylim(0,100)
plt.show()
hx = sns.barplot(y=Ref_eff_dataframe['Hidden Reference Identified %'].astype(float), x='Listener', hue='Attribute',
                 data=Ref_eff_dataframe)
sns.move_legend(hx, "upper left", bbox_to_anchor=(1, 1))
handles, _ = hx.get_legend_handles_labels()
handles.append(plt.axhline(y=80, c='red', linestyle='dashed', label="80% threshold"))
plt.legend()
plt.xlabel('Listener ID')
plt.ylim(0,100)
plt.show()
#%% Remove non-reliable subjects
# listeners_remove = [2,3,7,8,11,12,14,15]
# index_remove = len(listeners_remove)
# presentation_order =np.array([np.delete(presentation_order,listeners_remove)])
# listeners =np.array([np.delete(listeners,listeners_remove)])

# for f in listeners_remove:
#     data_label_melt=data_label_melt[data_label_melt.Listener != int(f)]
#     dataBAQ_label_melt=dataBAQ_label_melt[dataBAQ_label_melt.Listener != int(f)]
#     listeners =np.delete(listeners,np.where(listeners == int(f)))


#%% Statistics summary
data_final = data_label_melt.append(dataBAQ_label_melt, ignore_index=True)
Summary_statistics_Env= data_label_melt['Rating'].agg(["count", "min", "max", "median", "mean", "skew"])
Summary_statistics_BAQ= dataBAQ_label_melt['Rating'].agg(["count", "min", "max", "median", "mean", "skew"])

###Shapiro's test (normality testq )
#Envelopment
for i in ["2.0","5.0","7.0.4"]:
    for k in ["Choir","Movie","Pop","Jazz"]:
        normality = data_label_melt[(data_label_melt['Channel'] == i) & (dataBAQ_label_melt['Excerpt']== k)]
        normality['Rating'] = (normality['Rating'] - normality['Rating'].min()) / (normality['Rating'].max() - normality['Rating'].min())
        result = shapiro(normality['Rating'])
        print("For Envelopment ratings with",i,"and for excerpt",k,"the p-value is = ",result.pvalue)
#BAQ
for i in ["2.0","5.0","7.0.4"]:
    for k in ["Choir","Movie","Pop","Jazz"]:
        normality = dataBAQ_label_melt[(dataBAQ_label_melt['Channel'] == i) & (dataBAQ_label_melt['Excerpt']== k)]
        normality['Rating'] = (normality['Rating'] - normality['Rating'].min()) / (normality['Rating'].max() - normality['Rating'].min())  
        result = shapiro(normality['Rating'])
        print("For BAQ ratings with ",i," and for excerpt",k,"the p-value is = ",result.pvalue)
plt.hist(data_label_melt['Rating'])
plt.xlabel('Envelopment')
plt.ylabel('Frequency')
plt.show()
plt.hist(dataBAQ_label_melt['Rating'])
plt.xlabel('BAQ')
plt.ylabel('Frequency')
plt.show()


#%% Box plots -  Exploratory analysis
sns.set_theme(style="whitegrid")
ax = sns.boxplot(y='Rating', x="Loudspeaker_layout",hue='Attribute',
                 data=data_final)
ax.axhline(y = 80,color = "red", linestyle = "dashed")
handles, _ = ax.get_legend_handles_labels()
handles.append(plt.axhline(y=80, c='green', linestyle='dashed', label="Reference"))
handles.append(plt.axhline(y=20, c='red', linestyle='dashed', label="Low Anchor"))
plt.legend()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.ylim(0,100)
plt.xlabel('Loudspeaker Displacement Type')
plt.ylabel('Attribute Rating')
plt.show()
#%% Participants rating behaviour/ Which excerpt did my participants missidentified the reference
sns.set_style("whitegrid")

###Envelopment
plt.figure()
ex = sns.boxplot(y='Rating', x='Listener',notch='true', data=data_label_melt)
ex.set_title('Participant ratings')
plt.show()

plt.figure()
y_Env=data_label_melt[(data_label_melt["Loudspeaker_layout"] == 'No Disp.') ]
ex = sns.boxplot(y='Rating', x='Listener', hue='Excerpt',data=y_Env)
ex.set_title('Hidden Reference ratings')
plt.ylabel('Env. Rating')
plt.legend(title="Programme Material")
plt.ylim(0,100)
plt.show()


###BAQ
plt.figure()
fx = sns.boxplot(y='Rating', x='Listener',notch='true',data=dataBAQ_label_melt)
fx.set_title('Participant ratings')
plt.show()

plt.figure()
y_BAQ=dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"] == 'No Disp.') ]
fx = sns.boxplot(y='Rating', x='Listener', hue='Excerpt',                  data=y_BAQ)
fx.set_title('Hidden Reference ratings')
plt.ylabel('BAQ Rating')
plt.legend(title="Programme Material")
plt.ylim(0,100)
plt.show()

#%% ANOVA model

y_Env_norm=data_label_melt
y_BAQ_norm=dataBAQ_label_melt
##ANOVA Envelopment##


# model1 = smf.ols("""Rating ~ -1 + C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
#                C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
#                C(Loudspeaker_layout):C(Excerpt):C(Channel)""", y_Env_norm,groups="Listener").fit()
model1 = smf.ols("""Rating ~ -1 + C(Loudspeaker_layout) + C(Channel) + C(Loudspeaker_layout):C(Channel)""", y_Env_norm,groups="Listener").fit()
# perform ANOVA
ANOVA_summary1 = sm.stats.anova_lm(model1, typ=2)
print(ANOVA_summary1)

sns.set_theme(style="whitegrid")

# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
fig1=plt.hist(model1.resid, bins='auto', histtype='bar', ec='k')
plt.title("Frequency of Residuals Env. model")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


#sns.pointplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Channel',errorbar='ci,95')
fig2= sms.graphics.factorplots.interaction_plot( x=data_label_melt['Loudspeaker_layout'], trace=data_label_melt['Channel'], response=data_label_melt['Rating'])
plt.legend(title="Loudspeaker Layout")
plt.xlabel('Loudspeaker Displacement Type')
plt.ylabel('Mean of Envelopment Rating')
plt.ylim([0,100])
plt.show()


## ANOVA BAQ##
# model2 = smf.ols("""Rating ~ -1 + C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
#                C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
#                C(Loudspeaker_layout):C(Excerpt):C(Channel)""", y_BAQ_norm,groups="Listener").fit()
model2 = smf.ols("""Rating ~ -1 + C(Loudspeaker_layout) + C(Channel) + C(Loudspeaker_layout):C(Channel)""", y_BAQ_norm,groups="Listener").fit()

# perform ANOVA
ANOVA_summary2 = sm.stats.anova_lm(model2, typ=2)
print(ANOVA_summary2)

sns.set_theme(style="whitegrid")

# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
fig3=plt.hist(model2.resid, bins='auto', histtype='bar', ec='k')
plt.title("Frequency of Residuals BAQ model")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

#sns.pointplot(data=data_label_melt,x='Loudspeaker_layout',y='Envelopment',hue='Channel',errorbar='ci,95')
fig4= sms.graphics.factorplots.interaction_plot( x=dataBAQ_label_melt['Loudspeaker_layout'], trace=dataBAQ_label_melt['Channel'], response=dataBAQ_label_melt['Rating'])
plt.legend(title="Loudspeaker Layout")
plt.xlabel('Loudspeaker Displacement Type')
plt.ylabel('Mean of BAQ Rating')
plt.ylim([0,100])
plt.show()

#%%post-hoc

tukey_Env = pairwise_tukeyhsd(endog=data_label_melt['Rating'], groups=data_label_melt['Loudspeaker_layout'], alpha=0.05)
print(tukey_Env)
tukey_Env = pairwise_tukeyhsd(endog=data_label_melt['Rating'], groups=data_label_melt['Excerpt'], alpha=0.05)
print(tukey_Env)
tukey_Env = pairwise_tukeyhsd(endog=data_label_melt['Rating'], groups=data_label_melt['Channel'], alpha=0.05)
print(tukey_Env)


tukey_BAQ = pairwise_tukeyhsd(endog=dataBAQ_label_melt['Rating'], groups=dataBAQ_label_melt['Loudspeaker_layout'], alpha=0.05)
print(tukey_BAQ)
tukey_BAQ = pairwise_tukeyhsd(endog=dataBAQ_label_melt['Rating'], groups=dataBAQ_label_melt['Excerpt'], alpha=0.05)
print(tukey_BAQ)
tukey_BAQ = pairwise_tukeyhsd(endog=dataBAQ_label_melt['Rating'], groups=dataBAQ_label_melt['Channel'], alpha=0.05)
print(tukey_BAQ)

#%% Linear mixed effect model 


###BAQ attribute

##Create the model
model_BAQ =smf.mixedlm("""Rating ~ -1 + C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", y_BAQ_norm,groups="Listener").fit()
print(model_BAQ.summary())

##Reduce model to significant effects
model_BAQ =smf.mixedlm("""Rating ~ -1 + C(Loudspeaker_layout)""", y_BAQ_norm,groups="Listener").fit()
print(model_BAQ.summary())
r1=model_BAQ.fe_params

##Plot the means
plt.figure() 
plt.errorbar(["Azimuthal Disp.","Distance Disp.","Combined Disp.","No Disp."],r1.array,yerr=2*model_BAQ.bse_fe,fmt='o');
plt.title("Linear mixed effect model means (Listener as random effect)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("BAQ mean rating")
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
model_Env =smf.mixedlm("""Rating ~ -1 + C(Loudspeaker_layout) + C(Excerpt) + C(Channel) +
               C(Loudspeaker_layout):C(Excerpt) + C(Loudspeaker_layout):C(Channel) + C(Excerpt):C(Channel) +
               C(Loudspeaker_layout):C(Excerpt):C(Channel)""", y_Env_norm,groups="Listener").fit()
print(model_Env.summary())
model_Env= smf.mixedlm("""Rating ~ -1 + C(Loudspeaker_layout)""", y_Env_norm,groups="Listener").fit()
print(model_Env.summary())
r2=model_Env.fe_params

##Plot the means
plt.figure() 
plt.errorbar(["Azimuthal Disp.","Distance Disp.","Combined Disp.","No Disp."],r2.array,yerr=2*model_Env.bse_fe,fmt='o');
plt.title("Linear mixed effect model means (Listener as random effect)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Envelopment mean rating")       
plt.show()
##plot residuals vs preddicted
plt.figure()
plt.scatter(model_Env.resid, model_Env.fittedvalues, alpha = 0.5)
plt.title("Residual vs. Fitted in Python")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.savefig('python_plot.png',dpi=300)
plt.show()
fig = plt.figure(figsize = (16, 9))
##Residuals kdistribution
plt.figure()
ax = sns.distplot(model_Env.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
ax.set_xlabel("Residuals")
##Q-Q PLot
fig2 = plt.figure(figsize = (16, 9))
ax = fig.add_subplot(111)
sm.qqplot(model_Env.resid, dist = stats.norm, line = 's', ax = ax)
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
c1=stats.pearsonr(dataBAQ_label_melt['Azimuth Error'],dataBAQ_label_melt.Rating)
print(c1)
##BAQ and Distance error
c2=stats.pearsonr(dataBAQ_label_melt['Distance Error'],dataBAQ_label_melt.Rating)
print(c2)
##Env and Azimuth error
c3=stats.pearsonr(data_label_melt['Azimuth Error'],data_label_melt.Rating)
print(c3)
##Env and Distance error
c4=stats.pearsonr(data_label_melt['Distance Error'],data_label_melt.Rating)
print(c4)


# Scatter plots of positional error
gx=sns.scatterplot(data=dataBAQ_label_melt,x='Loudspeaker_layout',y='Distance Error',hue='Channel')
plt.xlabel('Loudspeaker Displacement Type')
plt.ylabel('Distance Displacement/Channel [m/channel]')
plt.legend(title='Loudspeaker Layout')
plt.show()

hx=sns.scatterplot(data=dataBAQ_label_melt,x='Loudspeaker_layout',y='Azimuth Error',hue='Channel')
plt.xlabel('Loudspeaker Displacement Type')
plt.ylabel('Azimuthal Displacement/Channel [deg/channel]')
plt.legend(title='Loudspeaker Layout')
plt.show()
