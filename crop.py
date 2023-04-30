# for data manipulation
import numpy as np
import pandas as pd
from sklearn import *

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for interactivity
from ipywidgets import interact

# Let's read the dataset
data = pd.read_csv("E:\\Semester-06 Project\\Optimizing Agricultural Production\\CropData.csv")

# Let's check the shape of the dataset
print("Shape of the Dataset is : ", data.shape)

# Let's check the head of the dataset
data.head()

# Let's check if there is any missing value present in the dataset
data.isnull().sum()

# Let's check the Crops present in the dataset
data['label'].value_counts()

#  Let's check the Summary for all the crops

print("Average ratio of Nitrogen in the soil : {0:.2f}".format(data['N'].mean()))
print("Average ratio of Phosphorus in the soil : {0:.2f}".format(data['P'].mean()))
print("Average ratio of Potassium in the soil : {0:.2f}".format(data['K'].mean()))
print("Average Temprature in Celsius : {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(data['humidity'].mean()))
print("Average pH value of the soil : {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))

# Let's check the summary statistics for each of the crop

# Let's check the summary statistics for each of the crop

def summary(crop):
    x = data[data['label'] == crop]

    print("----------------------------------------")
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen required : ", x['N'].min())
    print("Average Nitrogen required : ", x['N'].mean())
    print("Maximum Nitrogen required : ", x['N'].max())

    print("----------------------------------------")
    print("Statistics for Phosphorus")
    print("Minimum Phosphorus required : ", x['P'].min())
    print("Average Phosphorus required : ", x['P'].mean())
    print("Maximum Phosphorus required : ", x['P'].max())

    print("----------------------------------------")
    print("Statistics for Potassium")
    print("Minimum Potassium required : ", x['K'].min())
    print("Average Potassium required : ", x['K'].mean())
    print("Maximum Potassium required : ", x['K'].max())

    print("----------------------------------------")
    print("Statistics for Temperature")
    print("Minimum Temperature required : {0:.2f}".format(x['temperature'].min()))
    print("Average Temperature required : {0:.2f}".format(x['temperature'].mean()))
    print("Maximum Temperature required : {0:.2f}".format(x['temperature'].max()))

    print("----------------------------------------")
    print("Statistics for Humidity")
    print("Minimum Humidity required : {0:.2f}".format(x['humidity'].min()))
    print("Average Humidity required : {0:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity required : {0:.2f}".format(x['humidity'].max()))

    print("----------------------------------------")
    print("Statistics for PH")
    print("Minimum PH required : {0:.2f}".format(x['ph'].min()))
    print("Average PH required : {0:.2f}".format(x['ph'].mean()))
    print("Maximum PH required : {0:.2f}".format(x['ph'].max()))

    print("----------------------------------------")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required : {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required : {0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required : {0:.2f}".format(x['rainfall'].max()))


# Let's compare average requirements for each crop with average conditions

@interact
def compare (conditions = ['N', 'P','K', 'temperature', 'ph', 'humidity', 'rainfall']):   
    print("Average Value for", conditions, "is {0:.2f}". format (data [conditions].mean()))
    print("-----------------------------------------------")
    print("Rice : {0:.2f}" .format(data[(data['label'] == 'rice')][conditions].mean()))
    print("Black Grams : {0:.2f}" .format(data[data[ 'label'] == 'blackgram'][conditions].mean()))
    print("Banana : {0:.2f}". format (data[ (data[ 'label'] == 'banana')] [conditions].mean()))
    print("Jute : {0:.2f}" .format(data[data['label'] == 'jute'][conditions].mean()))
    print("Coconut : {0:.2f}". format (data[ (data['label'] == 'coconut')] [conditions].mean()))
    print("Apple : {0:.2f}". format (data[data['label'] == 'apple'][conditions].mean()))
    print("Papaya : {0:.2f}". format (data[ (data[ 'label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}". format (data[data['label'] == 'muskmelon'][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[ (data[ 'label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}". format (data [data[ 'label'] == 'watermelon'][conditions].mean()))
    print("Kidney Beans : {0:.2f}".format(data[(data[ 'label'] == 'kidneybeans')] [conditions].mean()))
    print("Mung Beans : {0:.2f}". format (data [data[ 'label'] == 'mungbean' ][conditions].mean()))
    print("Oranges : {0:.2f}". format (data[(data[ 'label'] == 'orange')][conditions].mean()))
    print("Chick Peas : {0:.2f}". format (data[data[ 'label'] == 'chickpea'][conditions].mean()))
    print("Lentils : {0:.2f}". format (data[ (data[ 'label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0:.2f}". format (data[data[ 'label'] == 'cotton'][conditions].mean()))
    print("Maize : {0:.2f}".format(data[(data['label'] == 'maize') ][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[(data['label'] == 'mothbeans')] [conditions].mean()))
    print("Pigeon Peas :  {0:.2f}" .format(data[(data['label'] == 'pigeonpeas')][conditions].mean()))
    print("Mango : {0:.2f}" .format(data[(data['label'] == 'mango')][conditions].mean()))
    print("Pomegranate :  {0:.2f}" .format(data[(data['label'] == 'pomegranate')][conditions].mean()))
    print("Coffee :  {0:.2f}" .format(data[(data['label'] == 'coffee')][conditions].mean()))

@interact
def compare (conditions = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print("Crops which require greater than average", conditions, '\n')
    print (data[data [conditions]>data[conditions].mean()]['label']. unique())
    
    print("--------------------------------")
    print("Crops which require less than average", conditions, '\n')
    print (data[data[conditions] <= data[conditions].mean()]['label']. unique())

plt.subplot(2,4,1)
sns.histplot(data['N'], color = 'yellow')
plt.xlabel('Ratio of Nitrogen', fontsize = 12)
plt.grid()
           
plt.subplot(2,4,2)
sns.histplot(data['P'], color = 'orange')
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
sns.histplot(data['rainfall' ], color = 'grey')
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

print("Some Interesting Patterns")
print("------------------------")
print("Crops which requires very High Ratio of Nitrogen Content in Soil:", data[data['N'] > 120] ['label'].unique())
print("Crops which requires very High Ratio of Phosphorous Content in Soil:", data[data['P'] > 100] ['label'].unique())
print("Crops which requires very High Ratio of Potassium Content in Soil:", data[data['K'] > 200] ['label'].unique())
print("Crops which requires very High Rainfall:", data[data['rainfall'] > 200] ['label' ].unique())
print("Crops which requires very Low Temperature :", data[data['temperature'] <10] ['label'].unique())
print("Crops which requires very High Temperature :", data [data['temperature'] > 40]['label'].unique ())
print("Crops which requires very Low Humidity:", data[data['humidity'] <20]['label'].unique())
print("Crops which requires very Low pH:", data [data['ph'] < 4]['label' ].unique())
print("Crops which requires very High pH:", data [data['ph'] > 9]['label']. unique())

### Lets understand which crops can only be Grown in Summer Season, Winter Season and Rainy Season

print("Summer Crops")
print (data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("------------------------")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label']. unique())

print("------------------------")
print("Rainy Crops")
print (data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())

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
    km = KMeans (n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)

# Lets plot the results

plt.plot(range (1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# Lets implement the K Means algorithm to perform Clustering analysis

km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

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

log_model = LogisticRegression()

log_model.fit(x_train, y_train)

y_pred = log_model.predict(x_test)

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

prediction = log_model.predict((np. array([[90,40,40,20,80,7,200]])))

print("The Suggested Crop for Given Climatic Condition is ", prediction)

data[data['label'] == 'coffee']

prediction = log_model.predict(np.array([[1110,13,15,25,60,8,1050]]))
"""a  = [[1110,13,15,25,60,8,1050]]
predict(a)"""
print("The Suggested Crop for Given Climatic Condition is ", prediction)

# def predict(df):
#    pred = log_model.predict(df)
# 	# return pred
#    print(pred)

# Creating a Pickle file using serialization
import pickle 
pickle_out = open("E:\\Semester-06 Project\\Optimizing Agricultural Production\\file.pkl" , "wb")
pickle.dump(km, pickle_out)
pickle_out.close()

km.predict([[90,40,40,20,80,7,200]])
















