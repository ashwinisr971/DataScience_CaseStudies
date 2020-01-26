import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('winequality-white.csv',sep=';')

#checking for missing values
msno.matrix(data,figsize=(10,3))

#Distribution
fig,axes=plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(10,20)
sns.boxplot(data=data,orient='v',ax=axes[0])
sns.boxplot(data=data,y='quality',orient='ph',ax=axes[1])

#correlation analysis
corrMatt=data.corr()
mask=np.array(corrMatt)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt,mask=mask,vmax=.8,square=True,annot=True)
