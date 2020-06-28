import codecademylib3_seaborn
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')

# Update sex column to numerical
passengers['Sex'] =  passengers['Sex'].map({'male':0,'female':1})


# Fill the nan values in the age column
passengers['Age'].fillna(inplace=True,value=round(passengers['Age'].mean()))
print(passengers)
# Create a first class column
passengers['FirstClass']=passengers['Pclass'].apply(lambda p: 1 if p==1 else 0)

passengers['SecondClass']=passengers['Pclass'].apply(lambda p: 1 if p==2 else 0)

print(passengers)

# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features,test_features,train_labels,test_labels = train_test.split(features,survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1

scalar = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
# Create and train the model
model = LogiticRegression()
model.fit(train_features,train_labels)

# Score the model on the train data
print(model.score(train_features,train_labels))

# Score the model on the test data


# Analyze the coefficients


# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
#You = np.array([___,___,___,___])

# Combine passenger arrays

sample_passengers = scaler.transform(sample_passengers)
# Scale the sample passenger features
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))

# Make survival predictions!

