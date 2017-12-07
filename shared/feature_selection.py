import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class FeatureRanker():

	def __init__(self, X, Y, classify):
		self.X = X
		self.Y = Y
		if classify:
			self.classification_ranking()
		else:
			self.regression_ranking()

	def ranking(ranks, names, order=1):
		print(ranks)
		return dict(zip(names, ranks))

	def classification_ranking(self):
		colnames = self.X.columns
		ranks = {}
		rfc = RandomForestClassifier(n_jobs=-1)
		rfc.fit(self.X, self.Y)
		ranks["RandomForest"] = dict(zip(colnames, rfc.feature_importances_))

		# Create empty dictionary to store the mean value calculated from all the scores
		r = {}
		for name in colnames:
			r[name] = round(np.mean([ranks[method][name] 
    			for method in ranks.keys()]), 2)
 
		ranks["Mean"] = r
		self.ranks = ranks
		return self

	def regression_ranking(self):
		# Define dictionary to store our rankings
		ranks = {}
		# Create our function which stores the feature rankings to the ranks dictionary

		colnames = self.X.columns
		# Finally let's run our Selection Stability method with Randomized Lasso
		rlasso = RandomizedLasso(alpha=0.04)
		rlasso.fit(self.X, self.Y)
		ranks["RandomizedLassoStability"] = self.ranking(np.abs(rlasso.scores_), colnames)

		# Recursive feature elimination 
		lr = LinearRegression(normalize=True)
		lr.fit(self.X,self.Y)
		#stop the search when only the last feature is left
		rfe = RFE(lr, n_features_to_select=1)
		rfe.fit(self.X,self.Y)
		ranks["RecursiveFeatureElimination"] = self.ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

		# Using Linear Regression
		lr = LinearRegression(normalize=True)
		lr.fit(self.X,self.Y)
		ranks["LinReg"] = self.ranking(np.abs(lr.coef_), colnames)

		# Using Ridge 
		ridge = Ridge(alpha = 7)
		ridge.fit(self.X,self.Y)
		ranks['Ridge'] = self.ranking(np.abs(ridge.coef_), colnames)

		# Using Lasso
		lasso = Lasso(alpha=.05)
		lasso.fit(self.X, self.Y)
		ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

		rf = RandomForestRegressor(n_jobs=-1, n_estimators=1000);
		rf.fit(self.X,self.Y)
		ranks["RandomForest"] = self.ranking(rf.feature_importances_, colnames)

		# Create empty dictionary to store the mean value calculated from all the scores
		r = {}
		for name in colnames:
			r[name] = round(np.mean([ranks[method][name] 
    			for method in ranks.keys()]), 2)
 
		ranks["Mean"] = r
		self.ranks = ranks
		return ranks

	def rank(self, by='RandomForest', plot=False):
		if by not in list(self.ranks.keys()):
			print('Available rankings', list(self.ranks.keys()))
			return;
		if plot:
			self.plot_ranking()
		return pd.DataFrame(list(self.ranks[by].items()), columns= ['Feature',by]).sort_values(by=by,ascending=False)

	'''
		ranking_name: ['RandomForest', 'Mean', 'Ridge', 'LinReg', 'Lasso', 'RecursiveFeatureElimination','RandomizedLassoStability']
	'''
	def plot_ranking(self, ranking_name='Mean'):
		ranking_df = pd.DataFrame(list(self.ranks[ranking_name].items()), columns= ['Feature',ranking_name])
		ranking_df = ranking_df.sort_values(ranking_name, ascending=False)
		ranking_df.plot(figsize=(15,8), kind='bar',x='Feature', y=ranking_name, colormap='plasma')
		plt.show()
