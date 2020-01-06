# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:43:32 2020

@author: SANJAY KUMAR
"""

#importing necessay libraries

import numpy as np
import pandas as pd
import seaborn as sns

#setting dimensions of the plot

sns.set(rc={'figure.figsize':(11.7,8.27)})

#reading csv file

cars_data=pd.read_csv('cars_sampled.csv')

#craeting a copy of the data
cars=cars_data.copy()

cars.info()

#summarizing data

cars.describe()
pd.set_option('display.float_format',lambda x: '%.3f' %x)
cars.describe()

pd.set_option('display.max_columns',500)
print(cars.describe())

#dropping unwanted columns
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)
print(cars.columns)


#Removing duplicates data
cars.drop_duplicates(keep='first',inplace=True)


#data cleaning

print(cars.isnull().sum())

#variable year of registration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)

sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)

#working range-1950-2019

#variable price
price_counts=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#working range 100 to 150000

#Variable PowerPS
power_counts=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#working range 10 to 500

#working range of data

cars=cars[
        (cars.yearOfRegistration<=2018)
        &(cars.yearOfRegistration>=1950)
        &(cars.price>=100)
        &(cars.price<=150000)
        &(cars.powerPS>=10)
        &(cars.powerPS<=500)]

cars['monthOfRegistration']/=12


# creating variable age

cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

#droppiong yearOfRegistration and monthOfRegistration

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualising parameters

sns.distplot(cars['Age'])
sns.boxplot(cars['Age'])

sns.distplot(cars['price'])
sns.boxplot(cars['price'])

sns.distplot(cars['powerPS'])
sns.boxplot(cars['powerPS'])


#age vs price

sns.regplot(x='Age',y='price',scatter='True',fit_reg=False,data=cars)
#price decreases as age increase and vice-versa
#however some cars are older and having high prices

#powerPS vs price

sns.regplot(x='powerPS',y='price',scatter='True',fit_reg=False,data=cars)

#variable seller

cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#insignificant variable seller

#variable offerType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'],columns='count',normalize=True)
sns.countplot(x='offerType',data=cars)
#insignificant variable

#variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
#equally distributed

sns.boxplot(x='abtest',y='price',data=cars)
#almost equal distribution, will not affect price that much

#variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)

sns.boxplot(x='vehicleType',y='price',data=cars)
#vehicleType affects price

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)

sns.boxplot(x='gearbox',y='price',data=cars)
#gearbox affects price

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)

sns.boxplot(x='model',y='price',data=cars)
#considering model

#variable kilometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=cars)
#considerig in modelling

#variable fueltype
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#fuelType affects price

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
#brand affects price

#variable notRepairedDamage
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)
#considering for modelling


#removing insignificant variables
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#checking correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlations=cars_select1.corr()
round(correlations,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


"""
We are going to build a linear regression model and random forest model on two sets of data
1.data obtained by omiiting the rows containing any missing values
2.data obtained by imputing the rows with any missing values

"""

cars_omit=cars.dropna(axis=0)

#converting categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)
 
#importing necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#model building with omitting data

#separating input and output features

x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

prices=pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()

y1=np.log(y1)

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)


#finding the mean for test data value
base_pred=np.mean(y_test)
print(base_pred)

#repeating same value till length of the test data
base_pred=np.repeat(base_pred,len(y_test))

#finding RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)



#Linear Regression with omitted data
lgr=LinearRegression(fit_intercept=True)

model_lin1=lgr.fit(x_train,y_train)

cars_predictions_lin1=lgr.predict(x_test)


#calculating MSE and RMSE
lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)


#R squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regession diagnostic- Residual plot analysis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residuals1, scatter=True, fit_reg=False, data=cars)
residuals1.describe()


#RandomForest model with omitted data

rf=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=100,min_samples_split=10,
                         min_samples_leaf=4,random_state=1)

model_rf1=rf.fit(x_train,y_train)

cars_predictions_rf1=rf.predict(x_test)

#computing MSE and RMSE
rf_mse1=mean_squared_error(y_test,cars_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#R squared value
r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)
print(r2_rf_test1,r2_rf_train1)




#Model builded with imputed data

cars_imputed=cars.apply(lambda x:x.fillna(x.median()) \
                        if x.dtype== 'float' else \
                        x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)

#model buliding with imputed data

x2=cars_imputed.drop(['price'],axis='columns',inplace=False)
y2=cars_imputed['price']

prices=pd.DataFrame({"1. Before":y2,"2. After":np.log(y2)})
prices.hist()

y2=np.log(y2)

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)


#finding the mean for test data value
base_pred=np.mean(y_test1)
print(base_pred)

#repeating same value till length of the test data
base_pred=np.repeat(base_pred,len(y_test1))

#finding RMSE
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1,base_pred))
print(base_root_mean_square_error_imputed)



#Linear Regression with omitted data
lgr2=LinearRegression(fit_intercept=True)

model_lin2=lgr.fit(x_train1,y_train1)

cars_predictions_lin2=lgr.predict(x_test1)


#calculating MSE and RMSE
lin_mse2=mean_squared_error(y_test1,cars_predictions_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)


#R squared value
r2_lin_test2=model_lin1.score(x_test1,y_test1)
r2_lin_train2=model_lin1.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

#Regession diagnostic- Residual plot analysis
residuals2=y_test1-cars_predictions_lin2
sns.regplot(x=cars_predictions_lin2, y=residuals2, scatter=True, fit_reg=False, data=cars)
residuals2.describe()



# Random Forest with imputed data

rf1=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=100,min_samples_split=10,
                         min_samples_leaf=4,random_state=1)

model_rf2=rf.fit(x_train1,y_train1)

cars_predictions_rf2=rf.predict(x_test1)

#computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1,cars_predictions_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#R squared value
r2_rf_test2=model_rf2.score(x_test1,y_test1)
r2_rf_train2=model_rf2.score(x_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)
