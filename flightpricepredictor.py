import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd

sns.set()

df = pd.read_excel(r"C:\Users\aniru\Desktop\pythonproject2\flightprice\Data_Train.xlsx")
pd.set_option('display.max_columns',None)

#observe the columns with respect to their data types
df.info()

#observe the number of records associated with duration
df["Duration"].value_counts()

#pre process

df.dropna(inplace=True)

#checking for null values
df.isnull().sum()

#EDA

df["journey_day"]= pd.to_datetime(df["Date_of_Journey"], format= r"%d/%m/%Y").dt.day
df["journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = r"%d/%m/%Y").dt.month

 # we no drop date of journey as we dont need it

df["Date_of_Journey"] = pd.DataFrame(df["Date_of_Journey"])
df.drop( axis =1, inplace= True, columns=["Date_of_Journey"])

#from departure time we want the hours and minutes in a manner similar to that of date of journey.

df["dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["dep_minute"] = pd.to_datetime(df["Dep_Time"]).dt.minute

df.drop(["Dep_Time"], axis =1, inplace=True)

#we do the same for the arrival time

df["Arr_hour"]= pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arr_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute

df.drop(["Arrival_Time"], axis = 1 , inplace = True)


#strip duration so that we may add 0m to the vales without it and 0h to values without it.

duration = list(df["Duration"])
for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if "h" in duration[i]:
            duration[i] = duration[i] + "0m"
        else:
            duration[i] = "0h" + duration[i]
duration_hours = []
duration_minutes = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))
    duration_minutes.append(int(duration[i].split(sep = "h")[-1].split(sep="m")[0]))

#create columns for duration hours and minutes

df["duration hours"]= duration_hours
df["duration minutes"] = duration_minutes

#drop duration

df.drop(["Duration"], axis =1, inplace=True)

#plotting a graph to compare the prices we see that Jetairways has the highest prices

sns.catplot(y = "Price", x= "Airline", data = df.sort_values("Price", ascending= False), kind = "boxen", height = 6, aspect=3)

#after having observed the plot we will now create dummy variables for the airlines accordingly.

Airline = df[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first=True)

sns.catplot(y = "Price", x= "Source", data = df.sort_values("Price", ascending= False), kind = "boxen", height = 6, aspect=3)

Source = df[["Source"]]
Source = pd.get_dummies(Source, drop_first=True)

Destination = df[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first=True)

#additional info contains 80% no info
#total stops and route are connected to each other

df.drop(["Route", "Additional_Info"], axis =1, inplace = True)

#we now perform labelencoding for the categorical types where values are assigned keys

df.replace({"non-stop" : 0,"1 stop" : 1,"2 stops" : 2, "3 stops":3, "4 stops":4}, inplace=True)

#   Concatenate Dataframe by combining all tthe new geatures from above

df1 = pd.concat([df, Airline, Source, Destination], axis=1)
#now

df1.drop(["Airline","Source","Destination"], axis =1, inplace = True)


#now we read in the test data set. in the test data set we do not have the dependent feature which is the price
#all other features are present in this test set.
#in the test data set we will repeat all of the pre processing steps as above.

df2 = pd.read_excel(r"C:\Users\aniru\Desktop\pythonproject2\flightprice\Test_set.xlsx")

sns.set()
pd.set_option('display.max_columns',None)

#observe the columns with respect to their data types
df2.info()

#observe the number of records associated with duration
df2["Duration"].value_counts()

#pre process

df2.dropna(inplace=True)

#checking for null values
df2.isnull().sum()

#EDA

df2["journey_day"]= pd.to_datetime(df2["Date_of_Journey"], format= r"%d/%m/%Y").dt.day
df2["journey_month"] = pd.to_datetime(df2["Date_of_Journey"], format = r"%d/%m/%Y").dt.month

 # we no drop date of journey as we dont need it

df2["Date_of_Journey"] = pd.DataFrame(df2["Date_of_Journey"])
df2.drop( axis =1, inplace= True, columns=["Date_of_Journey"])

#from departure time we want the hours and minutes in a manner similar to that of date of journey.

df2["dep_hour"] = pd.to_datetime(df2["Dep_Time"]).dt.hour
df2["dep_minute"] = pd.to_datetime(df2["Dep_Time"]).dt.minute

df2.drop(["Dep_Time"], axis =1, inplace=True)

#we do the same for the arrival time

df2["Arr_hour"]= pd.to_datetime(df2["Arrival_Time"]).dt.hour
df2["Arr_min"] = pd.to_datetime(df2["Arrival_Time"]).dt.minute

df2.drop(["Arrival_Time"], axis = 1 , inplace = True)

#strip duration so that we may add 0m to the vales without it and 0h to values without it.

duration = list(df2["Duration"])
for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if "h" in duration[i]:
            duration[i] = duration[i] + "0m"
        else:
            duration[i] = "0h" + duration[i]
duration_hours = []
duration_minutes = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))
    duration_minutes.append(int(duration[i].split(sep = "h")[-1].split(sep="m")[0]))

#create columns for duration hours and minutes

df2["duration hours"]= duration_hours
df2["duration minutes"] = duration_minutes

#drop duration

df2.drop(["Duration"], axis =1, inplace=True)


Airline = df2[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first=True)

Source = df2[["Source"]]
Source = pd.get_dummies(Source, drop_first=True)

Destination = df2[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first=True)

#additional info contains 80% no info
#total stops and route are connected to each other

df2.drop(["Route", "Additional_Info"], axis =1, inplace = True)

#we now perform labelencoding for the categorical types where values are assigned keys

df2.replace({"non-stop" : 0,"1 stop" : 1,"2 stops" : 2, "3 stops":3, "4 stops":4}, inplace=True)

#   Concatenate Dataframe by combining all tthe new geatures from above

df3 = pd.concat([df2, Airline, Source, Destination], axis=1)
#now

df3.drop(["Airline","Source","Destination"], axis =1, inplace = True)

#feature selection: we will now fit our datasets into a model to predict the dependency

#print columns for feature selection

df1.columns

#load columns into a variable X. we ensure that the dependent feature is not present

X1 = df1.loc[:,['Total_Stops', 'journey_day', 'journey_month', 'dep_hour',
       'dep_minute', 'Arr_hour', 'Arr_min', 'duration hours',
       'duration minutes', 'Airline_Air India', 'Airline_GoAir',
       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]

#we now load the dependent feature into a variable y

Y1 = df1.loc[:,["Price"]]

#plot heatmap to determine the correlation between the independent and dependent variables.

plt.figure(figsize=(44,28))
sns.heatmap(df1.corr(), annot=True, cmap='RdYlGn')



#from the correlation map if we observe that one more of the independent features are highly correlated(say 80%>)
# then we may drop any one of those two indepedent features to improve our model.

#we see here that total stops are the most important feature.

#we now fit our model into the ExtraTressRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X1,Y1)

#this helps us determine the important features that contribute the most to the dependency Y1

plt.figure(figsize=(12,18))
feat_importances = pd.Series(selection.feature_importances_, index=X1.columns)
#feat_importances.nlargest(20).plot(kind = 'barh')

#FITTING OUR MODEL INTO A RANDOM FOREST CLASSIFIER

#train_test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X1,Y1,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(x_train,y_train)

y_pred = reg_rf.predict(x_test)

reg_rf.score(x_train,y_train)
reg_rf.score(x_test,y_test)

#will have to verify the displot in sns once more

#we will now try implementing a scatter plot

plt.scatter(y_test,y_pred,alpha =0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")

#we now obtain metrics

from sklearn import metrics
metrics.r2_score(y_test,y_pred)



#HYPERPARAMETER TUNING

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop=1200, num =12)]
#number of features to consider at every split
max_features = ['auto','sqrt']
#maxinum number of levels in a tree
max_depth = [int(x) for x in np.linspace(5,30,num =6)]
#min number of samples required to split at a node
min_samples_leaf =[1,2,5,10]

random_grid = {'n_estimators': n_estimators, 'max_features':max_features, 'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf}

#random search using  fold cross validation. search across a 100 samples
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,
                               scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(x_train,y_train)
#save the model to be used again.
import joblib

import pickle,gzip


with gzip.open('flight_rf.pkl','wb') as file:
     joblib.dump(rf_random, file)

with open('flight_rf.pkl','rb') as model:
  forest=   joblib.load(model,'rb')
y_prediction = forest.predict(x_test)
metrics.r2_score(y_test, y_prediction)