# for data manipulation
import numpy as np
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("CropData.csv")

#  Let's check the Summary for all the crops

# print("Average ratio of Nitrogen in the soil : {0:.2f}".format(data['N'].mean()))
# print("Average ratio of Phosphorus in the soil : {0:.2f}".format(data['P'].mean()))
# print("Average ratio of Potassium in the soil : {0:.2f}".format(data['K'].mean()))
# print("Average Temprature in Celsius : {0:.2f}".format(data['temperature'].mean()))
# print("Average Relative Humidity in % : {0:.2f}".format(data['humidity'].mean()))
# print("Average pH value of the soil : {0:.2f}".format(data['ph'].mean()))
# print("Average Rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))

# Let's check the summary statistics for each of the crop

def summary(crop_name) :
    x = data[data['label'] == crop_name]
    frame = {
        "Labels" : ["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"],
        "Minimum" : [x["N"].min() , x["P"].min(),x["K"].min() , x["temperature"].min(),x["humidity"].min(), x["ph"].min(),x["rainfall"].min()],
        "Average" : [x["N"].mean() , x["P"].mean(),x["K"].mean() , x["temperature"].mean(),x["humidity"].mean(), x["ph"].mean(),x["rainfall"].mean()],
        "Maximum" : [x["N"].max() , x["P"].max(),x["K"].max() , x["temperature"].max(),x["humidity"].max(), x["ph"].max(),x["rainfall"].max()]
    }

    df = pd.DataFrame(frame)
    return df

def suggest_season(crop_name, data):
    if (data[(data['label'] == crop_name) & (data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique()).any():
        return "Summer"
    elif (data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique()).any():
        return "Winter"
    elif (data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique()).any():
        return "Rainy"
    else:
        return "doesn't fall under any seasonal category"

plt.subplot(2,4,1)
sns.histplot(data['N'], color = 'brown')
plt.xlabel('Ratio of Nitrogen', fontsize = 12)
plt.grid()
           
plt.subplot(2,4,2)
sns.histplot(data['P'], color = 'lightblue')
plt.xlabel('Ratio of Phosphorus', fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.histplot(data['K'], color = 'darkblue')
plt.xlabel('Ratio of Potassium', fontsize = 12)
plt.grid()
           
plt.subplot(2, 4, 4)
sns.histplot(data['temperature'], color = 'black')
plt.xlabel('Temperature', fontsize = 12)
plt.grid()
             
plt.subplot(2, 4, 5)
sns.histplot (data['rainfall' ], color = 'grey')
plt.xlabel('Rainfall', fontsize = 12)
plt.grid()
              
plt.subplot(2, 4, 6)
sns.histplot(data['humidity'], color = 'lightgreen')
plt.xlabel('Humidity', fontsize = 12)
plt.grid()
             
plt.subplot (2, 4, 7)
sns.histplot (data['ph'], color = 'darkgreen')
plt.xlabel('pH Level', fontsize = 12)
plt.grid()
              
plt.suptitle("Distribution for Agricultural Condition : ", fontsize = 20)
plt.show()

## Lets find out some Interesting Facts
def knowing_facts(data) :

    crop_list = []

    for crop1 in data[data['N'] > 120]['label'].unique() :
        crop_list.append(crop1)

    for crop2 in data[data['P'] > 100]['label'].unique() :
        crop_list.append(crop2)

    for crop3 in data[data['K'] > 200]['label'].unique() :
        crop_list.append(crop3)

    for crop4 in data[data['rainfall'] > 200]['label'].unique() :
        crop_list.append(crop4)

    for crop5 in data[data['temperature'] < 10]['label'].unique() :
        crop_list.append(crop5)

    for crop6 in data[data['temperature'] > 40]['label'].unique() :
        crop_list.append(crop6)

    for crop7 in data[data['humidity'] < 20]['label'].unique() :
        crop_list.append(crop7)

    for crop8 in data[data['ph'] > 9]['label'].unique() :
        crop_list.append(crop8)

    return crop_list

# Importing KMeans Clustering Library

from sklearn.cluster import KMeans

#removing the Labels column
x = data.drop(['label'], axis=1)

#selecting all the values of the data
x = x.values

# checking the shape
print(x.shape)

# Lets determine the Optimum Number of Clusters within the Dataset

plt.rcParams['figure.figsize'] = (10, 4)
wcss = []
for i in range(1, 11):
    km_model = KMeans (n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km_model.fit(x)
    wcss.append(km_model.inertia_)

# Lets plot the results

plt.plot(range (1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

# Lets implement the K Means algorithm to perform Clustering analysis

km_model = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km_model.fit_predict(x)

#Lets find out the Results

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename (columns = {0: 'cluster'})

# Lets check the Clusters of each Crops
print("Lets check the Results After Applying the K Means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())

print("------------------------------------------------------")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())

print("------------------------------------------------------")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())

print("-------------------------------------------------------")
print("Crops in Forth Cluster:", z[z['cluster'] == 3]['label'].unique())

# Lets split the Dataset for Predictive Modelling
y = data[ 'label']
x = data.drop(['label'], axis = 1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

# Lets create Training and Testing Sets for Validation of Results

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size = 0.2, random_state = 0)

print("The Shape of x train:", x_train.shape)
print("The Shape of x test:", x_test.shape)

print("The Shape of y train:", y_train.shape)
print("The Shape of y test:", y_test.shape)

# Lets create a Predictive Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict (x_test)

# Lets evaluate the Model Performance

from sklearn.metrics import confusion_matrix

# Lets print the Confusion matrix first

plt.rcParams['figure.figsize'] = (10, 10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'Wistia')
plt.title('Confusion Matrix for Logistic Regression', fontsize = 15)
plt.show()

# Lets print the Classification Report also
from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)
print(cr)

# Let's check the head of the dataset
data.head()

# prediction = model.predict((np. array([[90,40,40,20,80,7,200]])))
#
# print("The Suggested Crop for Given Climatic Condition is ", prediction)

# prediction = model.predict(np. array([[111,13,15,25,60,8,105]]))
# print("The Suggested Crop for Given Climatic Condition is ", prediction)

# Creating a Pickle file using serialization

import pickle 
pickle_out = open("km_model.pkl" , "wb")
pickle.dump(km_model , pickle_out)
pickle_out.close()

km_model.predict([[90,40,40,20,80,7,200]])




