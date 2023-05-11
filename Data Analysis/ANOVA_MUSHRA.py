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
import prince
from pandas.io.formats import style

#print(f"File location using __file__ variable: {os.path.realpath(os.path.dirname('ANOVA_MUSHRA.py'))}")print(f"File location using __file__ variable: {os.path.realpath(os.path.dirname('ANOVA_MUSHRA.py'))}")
os.chdir(r"C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis") # absolute path, using \ and r prefix
labels = np.loadtxt('naming.txt',dtype=str,delimiter=',', usecols =(0,1,2)) #load txt files to label all values. To fit for a stats model

###read the .txt files and create a data frame - Envelopment###
os.chdir(r'C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis\Envelopment') # absolute path, using \ and r prefix
filenames = [i for i in glob.glob("*.txt")];
participants = len(filenames);
listeners =np.arange(0,participants);
dataEnv = [np.loadtxt(file, skiprows=1,delimiter=' ',max_rows=48,usecols =(3), 
 unpack = True)
  for file in filenames]

      #reshape matrix and assign columns names
dataf = pd.DataFrame(np.transpose(np.array(dataEnv)));
data_label=dataf.assign(channel=labels[:,0], excerpt=labels[:,1],Loudspeaker_layout=labels[:,2]);

      #reshape the d dataframe suitable for statsmodels package using pd.melt
var = np.linspace(0,(len(filenames)-1),(len(filenames)));
data_label_melt = pd.melt(data_label, id_vars=['Loudspeaker_layout','channel','excerpt'], value_vars=var, value_name='Envelopment');

data_label_melt.rename(columns = {'variable':'Listener'}, inplace = True);




###create statistical model using statsmodel library###
model = ols("""Envelopment ~ C(Loudspeaker_layout) + C(excerpt) + C(channel) +
               C(Loudspeaker_layout):C(excerpt) + C(Loudspeaker_layout):C(channel) + C(excerpt):C(channel) +
               C(Loudspeaker_layout):C(excerpt):C(channel)""", data=data_label_melt).fit()
  ##perform ANOVA
ANOVA_summary1=sm.stats.anova_lm(model, typ=2)
print(ANOVA_summary1)

  ##Create boxplot per independent variable
ax = sns.boxplot(y='Envelopment',x="Loudspeaker_layout",  data=data_label_melt)
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

dx = sns.boxplot(y='Envelopment',x='excerpt',hue="Loudspeaker_layout",  data=data_label_melt)
#ax = sns.swarmplot(y='Envelopment',x='Loudspeaker_layout', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_Envelopment_excerpt-layout.png')
plt.show()

ex = sns.boxplot(y='Envelopment',x='excerpt',hue='channel', data=data_label_melt)
#bx = sns.swarmplot(y='Envelopment',x='excerpt', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_Envelopment_excerpt-channel.png')
plt.show()
    
###read the .txt files and create a data frame - BAQ###
os.chdir(r'C:\Users\feig\OneDrive - Bang & Olufsen\Next Generation Audio SA\Special Course\Three Driver Speaker arrangement to improve spatial awareness\Development\Special-Project\Result-Analysis\Data Analysis\BAQ') # absolute path, using \ and r prefix
filenames = [i for i in glob.glob("*.txt")]
dataBAQ = [np.loadtxt(file, skiprows=1,delimiter=' ',max_rows=48,usecols =(3), 
 unpack = True)
  for file in filenames]
#reshape matrix and assign columns names
dataBAQf = pd.DataFrame(np.transpose(np.array(dataBAQ)));
dataBAQ_label=dataBAQf.assign(channel=labels[:,0], excerpt=labels[:,1],Loudspeaker_layout=labels[:,2])
# reshape the d dataframe suitable for statsmodels package 
var = np.linspace(0,(len(filenames)-1),(len(filenames)))
dataBAQ_label_melt = pd.melt(dataBAQ_label, id_vars=['Loudspeaker_layout','channel','excerpt'], value_vars=var, value_name='BAQ')
dataBAQ_label_melt.rename(columns = {'variable':'Listener'}, inplace = True)


  ##create statistical model using statsmodel library
model = ols("""BAQ ~ C(Loudspeaker_layout) + C(excerpt) + C(channel) +
               C(Loudspeaker_layout):C(excerpt) + C(Loudspeaker_layout):C(channel) + C(excerpt):C(channel) +
               C(Loudspeaker_layout):C(excerpt):C(channel)""", data=dataBAQ_label_melt).fit()

  ##perform ANOVA
ANOVA_summary2=sm.stats.anova_lm(model, typ=2)
print(ANOVA_summary2)

  ##Create boxplot per independent variable
ax = sns.boxplot(y='BAQ',x='Loudspeaker_layout', data=dataBAQ_label_melt,color='#99c2a2')
ax = sns.swarmplot(y='BAQ',x='Loudspeaker_layout', data=dataBAQ_label_melt, color='#7d0013')
plt.savefig('boxplot-BAQ-Loudspeaker_layout.png')
plt.show()
bx = sns.boxplot(y='BAQ',x='excerpt', data=dataBAQ_label_melt,color='#99c2a2')
bx = sns.swarmplot(y='BAQ',x='excerpt', data=dataBAQ_label_melt, color='#7d0013')
plt.savefig('boxplot-BAQ-excerpt.png')
plt.show()
cx = sns.boxplot(y='BAQ',x='channel', data=dataBAQ_label_melt,color='#99c2a2')
cx = sns.swarmplot(y='BAQ',x='channel', data=dataBAQ_label_melt, color='#7d0013')
plt.savefig('boxplot-BAQ-channel.png')
plt.show()

dx = sns.boxplot(y='BAQ',x='excerpt',hue="Loudspeaker_layout",  data=dataBAQ_label_melt)
#ax = sns.swarmplot(y='Envelopment',x='Loudspeaker_layout', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_BAQ_excerpt-layout.png')
plt.show()
ex = sns.boxplot(y='BAQ',x='excerpt',hue='channel', data=dataBAQ_label_melt)
#bx = sns.swarmplot(y='Envelopment',x='excerpt', data=data_label_melt, color='#7d0013')
plt.savefig('boxplot2_BAQ_excerpt-channel.png')
plt.show()


###How many references were a correct hit during the test

##Envelopment
HitReference_Env=len(data_label_melt[(data_label_melt["Loudspeaker_layout"]=="version1") & 
             (data_label_melt["Envelopment"]==100)]);
Ref_eff_Env = HitReference_Env/(participants*12);

HitReference_Env_p=np.array([len(data_label_melt[(data_label_melt["Loudspeaker_layout"]=="version1") & 
             (data_label_melt["Envelopment"]==100)& (data_label_melt["Listener"]==listener)])
                                        for listener in listeners])/12;
Ref_eff_Env = HitReference_Env/(participants*12);

print( Ref_eff_Env,'were a correct hit in the Env test by the participants')

##BAQ
HitReference_BAQ=len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"]=="version1") & 
             (dataBAQ_label_melt["BAQ"]==100)]);
Ref_eff_BAQ = HitReference_BAQ/(participants*12);

HitReference_BAQ_p=np.array([len(dataBAQ_label_melt[(dataBAQ_label_melt["Loudspeaker_layout"]=="version1") & 
             (dataBAQ_label_melt["BAQ"]==100)& (dataBAQ_label_melt["Listener"]==listener)])
                                        for listener in listeners])/12;
print( Ref_eff_BAQ,'were a correct hit in the BAQ test by the participants')
Ref_eff_Env = HitReference_BAQ/(participants*12);
## Total
Ref_eff = (HitReference_Env+HitReference_BAQ)/(participants*12*2);

### Variance Analysis using prince package###

famd = prince.FAMD(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)
##BAQ
famd = famd.fit(data_label_melt);
BAQ_PC = famd.eigenvalues_summary;
var_att= np.array(famd.column_contributions_())*100