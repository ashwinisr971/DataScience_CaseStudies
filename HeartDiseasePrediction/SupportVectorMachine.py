import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal', 'heartdisease']
clivelandData = pd.read_csv('cleveland.csv', names = features)
hungarianData = pd.read_csv('hungary.csv', names = features)
switzerlandData = pd.read_csv('switzerland.csv', names = features)

datatemp = [clivelandData, hungarianData, switzerlandData]
data = pd.concat(datatemp)

#Preprocessing data
data=data.drop(["ca","slope","thal",],axis=1)
data=data.replace('?',np.nan)
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imputedData=imp.fit_transform(data)
print(type(imputedData))

#msno.matrix(pd.DataFrame(imputeData),figsize=(10,3))

X_train, X_test, y_train, y_test = train_test_split(imputedData[:, :-1], 
                imputedData[:, -1], test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the Model
classifier = svm.SVC(kernel='rbf')
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

# Accuracy of predictions
print(accuracy_score(y_test, preds))

