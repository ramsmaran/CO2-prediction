#task 1 B
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics

#task 1 B
emission=pd.read_csv('CO2 emission.csv') #reading a csv file
emission

#task 2 A
emission.shape # to obtain the shape of the dateframe

  #task 2 B
emission.loc[0:4] #to obtain the top five rows of the dataframe

#task 2 C
emission_new=emission.iloc[:,-5:]  
emission_new

#task 3 A
emission.median() #median

#task 3 A
emission.describe() #describe includes count ,mean ,std, min, max and quartiles, and 50% quartile is median

#task 3 B  dtypes gets all the datatypes for every colums
emission.dtypes

#task 4 A creating a box and hist plots
for i in ['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']:
    plt.boxplot(emission[i])
    plt.show()
    
#task 4 A
for i in ['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']:
    plt.hist(emission_new[i])
    plt.show()
    
    #task 4 B
x=emission_new.corr()  #correlation between selected 4 columns
x

#task 4 C
sns.heatmap(x) #heat map of emission_new dataframe

#task 4 D
a=emission['ENGINESIZE']
b=emission['CO2EMISSIONS']
c=emission['FUELCONSUMPTION_COMB']
d=emission['CYLINDERS']
plt.subplot(1, 3, 1)
plt.scatter(a,b) #scatter plot between enginesize and co2 emissions
plt.subplot(1,3,2)
plt.scatter(d,b) #scatter plot between cylinders and co2 emissions
plt.subplot(1,3,3)
plt.scatter(c,b) #scatter plot fuel consumption comb and co2 emissions
#co2 emissions act as a dependent variable and the best columns that fit for linear regressions are the columns between cylinders and co3 emissions
#ENGINESIZE, FUELCONSUMPTION_COMB are the two independent variables 

#task 4 E
g = sns.lmplot(x='ENGINESIZE', y='CO2EMISSIONS', data=emission) #lmplot between enginesize and co2emissions

h = sns.lmplot(x='FUELCONSUMPTION_COMB', y='CO2EMISSIONS', data=emission) #lmplot between fuelconsumption_comb and co2emissions

#task 5 A
from sklearn.model_selection import train_test_split
x=emission['ENGINESIZE']
y=emission['CO2EMISSIONS']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.2)
model=linear_model.LinearRegression()

train_x=np.array(train_x).reshape(-1,1) #reshaping the 1d arrays to 2d array
train_y=np.array(train_y).reshape(-1,1)#reshaping 1d to 2d
model.fit(train_x,train_y)

model.coef_ #returns the slope of the line 
model.intercept_ #returns the constant of the line
train_x[:2]

pred_y=model.predict(train_x) #model predicted values
#task 5 B-1
plt.scatter(train_x, train_y,  color='red')#scatter plot between engine size and co2 emmision
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")
plt.show()

#task 5 B-2
plt.scatter(train_x, train_y,  color='red')#scatter plot with regression lines
plt.plot(train_x,pred_y)
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")
plt.show()


#task 5 C
plt.scatter(train_y,pred_y) #scatter plot between actual and predicted values of co2 emission
plt.xlabel("CO2 Emission")
plt.ylabel("Predicted CO2 Emission")
plt.show()


#task 5 D
from sklearn.metrics import mean_squared_error
mean_squared_error(train_y,pred_y) #finds the mean squared error between the actual and the predicted values

#task 6 A
mul_x=emission[['ENGINESIZE','CYLINDERS']]
y=emission['CO2EMISSIONS']
mul_y=np.array(y).reshape(-1,1)
mul_model=linear_model.LinearRegression()
mul_x,mul_test_x,mul_y,mul_test_y=train_test_split(mul_x,mul_y,train_size=.8)

mul_model.fit(mul_x,mul_y)
predmul_y=mul_model.predict(mul_x)
predmul_y
#task 6 B
plt.scatter(mul_y,predmul_y)
plt.xlabel("CO2 Emission")
plt.ylabel("Predicted CO2 Emission")
plt.show()

#task 6 C
from sklearn.metrics import mean_squared_error
mean_squared_error(mul_y,predmul_y) #finds the mean squared error between the actual and the predicted values
from sklearn.metrics import mean_squared_error
mean_squared_error(train_y,pred_y) #univariant

#task 6 D
#mean squared error of linear regression is greater than that of multivariate regression so multi variate model is better than linear regression
model.score(train_x,train_y) #linear model
mul_model.score(mul_x,mul_y)# multivariate model

#Multivariate model has the higher score than that of the linear regression model
#Hence it is multivariate model is better than that of linear model
