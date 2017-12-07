import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os

def percent_missing(X):
	total = X.isnull().sum().sort_values(ascending=False)
	percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
	return missing_data[missing_data['Total'] > 0]