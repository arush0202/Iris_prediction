# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'improved_iris_app.py'.
# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating a Logistic Regression model.
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

# S2.2: Copy the following code in the 'improved_iris_app.py' file after the previous code.
# Create a function that accepts an ML mode object say 'model' and the four features of an Iris flower as inputs and returns its name.
@st.cache()
def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
  species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

# S2.3: Copy the following code and paste it in the 'improved_iris_app.py' file after the previous code.
# Add title widget
st.sidebar.title("Iris Species Prediction App")

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_length = st.sidebar.slider('Sepal Length',iris_df['SepalLengthCm'].min(), iris_df['SepalLengthCm'].max())
s_width = st.sidebar.slider('Sepal Width',iris_df['SepalWidthCm'].min(), iris_df['SepalWidthCm'].max() )

p_length = st.sidebar.slider('Petal Length',iris_df['PetalLengthCm'].min(), iris_df['PetalLengthCm'].max() )
p_width = st.sidebar.slider('Petal Width',iris_df['PetalWidthCm'].min(), iris_df['PetalWidthCm'].max() )



# Add a select box in the sidebar with the 'Classifier' label.
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
# Also pass 3 options as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in the 'classifier' variable.

# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.

# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable.
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button("Predict"):
  if classifier == 'Support Vector Machine':
    species_type = prediction(svc_model, s_length,s_width,p_length,p_width)
    score = svc_model.score(X_train, y_train)
  elif classifier =='Logistic Regression':
    species_type = prediction(log_reg, s_length,s_width,p_length,p_width)
    score = log_reg.score(X_train, y_train)
  else:
    species_type = prediction(rf_clf, s_length,s_width,p_length,p_width)
    score = rf_clf.score(X_train, y_train)
    
  st.write("Species predicted:", species_type)
  st.write("Accuracy score of this model is:", score)