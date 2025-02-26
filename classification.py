import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
# Importing the dataset: the Iris dataset contains data of three spicies of floweras
dataset = load_iris()

# creating the DataFrame
data = pd.DataFrame(data = dataset.data, columns= dataset.feature_names)
data['target'] = dataset.target

# visualizing the first rows of the dataset
print(f"\nHere are the first 5 rows of the dataset:\n{data.head()}")

# Separete the data in features and target
x = data.iloc[:, : -1].values # all the columns except the last one
y = data["target"].values # the last column

# Splitting the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101, stratify= y)
# note: the "stratify" parameter ensures that classes are well balanced between train and test

# Feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) and NOT the y!
X_train_scaled = scaler.fit_transform(X_train) # fitting to X_train and transforming them
X_test_scaled = scaler.transform(X_test) # transforming X_test. DO NOT FIT THEM!


# --- AND OF MAIN CODE ---
