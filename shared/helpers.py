import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
from sklearn.preprocessing import LabelEncoder

def nc_features(X):
	categorical_features = X.select_dtypes(include = ["object"]).columns
	numerical_features = X.select_dtypes(exclude = ["object"]).columns
	return [numerical_features, categorical_features]

def label_encode(X, features=None):
	X_copy = X.copy()
	if features is None:
		features = X.columns

	for column in features:
		le = LabelEncoder()
		X_copy[column] = le.fit_transform(X_copy[column])

	return X_copy