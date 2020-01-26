import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal', 'heartdisease']

clivelanddata=pd.read_csv('cleveland.csv',names=features)
hungariandata=pd.read_csv('hungary.csv',names=features)
switzerlanddata=pd.read_csv('switzerland.csv',names=features)

datatemp=[clivelanddata,hungariandata,switzerlanddata]
data=pd.concat(datatemp)

data=data.drop(["ca","slope","thal"],axis=1)
data=data.replace('?',np.nan)

#missing data detection
msno.matrix(data,figsize=(10,3))


#outliers detection
fig,axes=plt.subplots(nrows=4,ncols=1)
fig.set_size_inches(10,20)
sns.boxplot(data=data,orient='v',ax=axes[0])
sns.boxplot(data=data,y='heartdisease',orient='v',ax=axes[1])
sns.boxplot(data=data,y='heartdisease',x='age',orient='v',ax=axes[2])
sns.boxplot(data=data,y='heartdisease',x='sex',orient='v',ax=axes[3])

#correlation analysis
corrMatt=data.corr()
mask=np.array(corrMatt)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots()
fig.set_size_inches(10,20)
sns.heatmap(corrMatt,mask=mask,vmax=.8,square=True,annot=True)


