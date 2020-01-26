import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

#importing data
data=pd.read_csv('winequality-white.csv',sep=';')
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
 
X=np.append(arr=np.ones((X.shape[0],1)),values=X,axis=1)

X_opt=X[:,[0,1,2,4,6,7,8,9,10,11]]

X_train,X_test,Y_train,Y_test=train_test_split(X_opt,Y)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Linear Regression
linearregression=LinearRegression()
linearregression.fit(X_train,Y_train)
prediction=linearregression.predict(X_test)
print(r2_score(Y_test,prediction))

X_opt=X[:,[0,1,2,4,6,7,8,9,10,11]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
print(regressor_OLS.summary())