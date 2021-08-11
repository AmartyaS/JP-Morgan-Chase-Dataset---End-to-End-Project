# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:32:03 2021

@author: ASUS
"""

# Importing all the required libraries
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from ml_metrics import rmse
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Reading the dataset
data=pd.read_csv(r"D:\Python Project\database.csv")

# Fills NA values
def navalues(value):
    value=value.fillna(value.mean())
    return value

# Exploratory Data Analysis
data.head()
data.columns
data.info()
data.isna().sum()
data.isnull().sum()
sns.heatmap(data.isna(),yticklabels=False,cbar=False) # Visualisation of NA values
data["Acquired Date"]=data["Acquired Date"].fillna(data["Acquired Date"].mode()[0])
data["Latitude"]=navalues(data["Latitude"])
data["Longitude"]=navalues(data["Longitude"])
data["2010 Deposits"]=navalues(data["2010 Deposits"])
data["2011 Deposits"]=navalues(data["2011 Deposits"])
data["2012 Deposits"]=navalues(data["2012 Deposits"])
data["2013 Deposits"]=navalues(data["2013 Deposits"])
data["2014 Deposits"]=navalues(data["2014 Deposits"])
data["2015 Deposits"]=navalues(data["2015 Deposits"])
corrmat=data.corr()
top_corr=corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(data[top_corr].corr(),annot=True)

# Age Calculator
def age(value):
    value=2021-value
    return value

# Data Manipulation
data["Established Date"]=pd.to_datetime(data["Established Date"])
data["Acquired Date"]=pd.to_datetime(data["Acquired Date"])
Established_Year=data["Established Date"].dt.year
Acquired_Year=data["Acquired Date"].dt.year
Acquired_Age=age(Acquired_Year)
Established_Age=age(Established_Year)
file=data[["Main Office","2010 Deposits","2011 Deposits","2012 Deposits","2013 Deposits","2014 Deposits",
         "2015 Deposits","2016 Deposits"]]
file["Established_Age"]=Established_Age
file["Acquired_Age"]=Acquired_Age
file.info()

                                                    

# Visualisation of Branch Counts Vs Acquired Age                                                            
sns.countplot(x="Acquired_Age",data=file,palette='bwr')
plt.ylabel("Count of Branches")

# Visualisation of Branch Counts Vs Acquired Age                                                            
sns.countplot(x="Established_Age",data=file,palette='bwr')
plt.ylabel("Count of Branches")

# Splitting dataset for train and test sample
x=file.drop(["2016 Deposits"],axis=1)
y=file["2016 Deposits"]

# Feature Importance
mod=ExtraTreesRegressor()
mod.fit(x,y)
mod.feature_importances_

# Plot graph of Important Features
feat_imp=pd.Series(mod.feature_importances_,index=x.columns)
feat_imp.nlargest(7).plot(kind='barh')
plt.show()

# Splitting the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# XGBoost Model
model=xgboost.XGBRegressor()
model.fit(x_train,y_train)
# Prediction
predictions_XG=model.predict(x_test)
# RMSE Score
rmse_xg=rmse(y_test,predictions_XG)

# RF model
rf_model=RandomForestRegressor()
rf_model.fit(x_train,y_train)
# Prediction
predictions_rf=rf_model.predict(x_test)
# RMSE Score
rmse_rf=rmse(y_test, predictions_rf)

#               *** Hyper-Parameter Tuning *** 
n_estimators=[int(i) for i in np.linspace(start=10, stop=1000, num=20)]
max_features=['auto','sqrt','log2']
max_depth=[int(i) for i in np.linspace(start=10,stop=1000,num=20)]
max_depth.append("None")
min_samples_split=[int(i) for i in np.linspace(start=2, stop=100,num=20)]
min_samples_leaf=[int(i) for i in np.linspace(start=1, stop=100,num=20)]

# RandomSearchCV Parameter List
random_grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth': max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf
             }
# Random-Forest with Hyperparameter Tuning - RandomSearchCV
rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,
                             n_iter=200,scoring='neg_mean_squared_error',cv=10,
                             verbose=2,n_jobs=1,random_state=42)
rf_random.fit(x_train,y_train)
# Predictions
predictions_random=rf_random.predict(x_test)
# Visualising the errors
sns.distplot(y_test-predictions_random)
plt.scatter(y_test,predictions_random)
# Best Parameters as per RandomSearchCV
rf_random.best_params_


# GridSearchCV Parameter List Based on RandomSearchCV's Best Parameters
n_estimators1=[int(i) for i in np.linspace(start=rf_random.best_params_['n_estimators']-20,
                                          stop=rf_random.best_params_['n_estimators']+20, num=5)]
max_features1=[rf_random.best_params_['max_features']]
max_depth1=[int(i) for i in np.linspace(start=rf_random.best_params_['max_depth']-10,
                                        stop=rf_random.best_params_['max_depth']+10,num=5)]
max_depth1.append("None")
min_samples_split1=[int(i) for i in np.linspace(start=rf_random.best_params_['min_samples_split']-10, 
                                                stop=rf_random.best_params_['min_samples_split']+10,num=5)]
min_samples_leaf1=[int(i) for i in np.linspace(start=rf_random.best_params_['min_samples_leaf'], 
                                               stop=rf_random.best_params_['min_samples_leaf']+10,num=5)]
# GridSearch Parameter Dictionary
grid={'n_estimators':n_estimators1,
             'max_features':max_features1,
             'max_depth': max_depth1,
             'min_samples_split':min_samples_split1,
             'min_samples_leaf':min_samples_leaf1
             }
# Random-Forest with Hyperparameter Tuning - GridSearchCV
rf=RandomForestRegressor()
rf_grid=GridSearchCV(estimator=rf,param_grid=grid,
                             scoring='neg_mean_squared_error',cv=10,
                             verbose=2,n_jobs=-1)
rf_grid.fit(x_train,y_train)
# Predictions
predictions_grid=rf_grid.predict(x_test)
# Visualising the errors
sns.distplot(y_test-predictions_grid)
plt.scatter(y_test,predictions_grid)


# Genetic Algorithm based on RandomSearchCV's Parameter List
tpot_re = TPOTRegressor(generations= 5, population_size= 500, offspring_size= 250,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestRegressor': random_grid}, 
                                 cv = 4)
tpot_re.fit(x_train,y_train)
# Predictions
predictions_tpot=tpot_re.predict(x_test)
# Visualising the Errors
sns.distplot(y_test-predictions_tpot)
plt.scatter(y_test,predictions)


# Plotting the Prediction Values of all the hyper-tuned models
frequency=[int(x) for x in np.linspace(start=10000, stop=10000000,num=10)]
plt.plot(y_test[0:10],frequency,label='Actual_Value')
plt.plot(predictions_tpot[0:10],frequency,label='TPOT_Predictions')
plt.plot(predictions_grid[0:10],frequency,label='GridSearchCV Predictions')
plt.plot(predictions_random[0:10],frequency,label='RandomSearchCV Predictions')
plt.plot(predictions_XG[0:10],frequency,label='XGBoost Predictions')
plt.legend()


# Open a file where you want to store the model
file=open('random_forest_grid_model.pkl','wb')
# Dump information to that file
pickle.dump(rf_grid,file)

# Loading the pickle file
model=pickle.load(open('random_forest_grid_model.pkl','rb'))