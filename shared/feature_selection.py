import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os

class FeatureRanker():

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.ranking_dict(self.X, self.Y)

	def ranking(self, name=None):
		if not name:
			return self.ranks
		return pd.DataFrame(list(self.ranks[name].items()), columns= ['Feature',name])

	def ranking_dict(self, X, Y):
		from sklearn.feature_selection import RFE, f_regression
		from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
		from sklearn.preprocessing import MinMaxScaler
		from sklearn.ensemble import RandomForestRegressor
		# Define dictionary to store our rankings
		ranks = {}
		# Create our function which stores the feature rankings to the ranks dictionary
		def ranking(ranks, names, order=1):
			minmax = MinMaxScaler()
			ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
			ranks = map(lambda x: round(x,2), ranks)
			return dict(zip(names, ranks))

		colnames = X.columns
		# Finally let's run our Selection Stability method with Randomized Lasso
		rlasso = RandomizedLasso(alpha=0.04)
		rlasso.fit(X, Y)
		ranks["RandomizedLassoStability"] = ranking(np.abs(rlasso.scores_), colnames)

		# Recursive feature elimination 
		lr = LinearRegression(normalize=True)
		lr.fit(X,Y)
		#stop the search when only the last feature is left
		rfe = RFE(lr, n_features_to_select=1)
		rfe.fit(X,Y)
		ranks["RecursiveFeatureElimination"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

		# Using Linear Regression
		lr = LinearRegression(normalize=True)
		lr.fit(X,Y)
		ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

		# Using Ridge 
		ridge = Ridge(alpha = 7)
		ridge.fit(X,Y)
		ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

		# Using Lasso
		lasso = Lasso(alpha=.05)
		lasso.fit(X, Y)
		ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)

		rf = RandomForestRegressor(n_jobs=-1, n_estimators=1000);
		rf.fit(X,Y)
		ranks["RandomForest"] = ranking(rf.feature_importances_, colnames)

		# Create empty dictionary to store the mean value calculated from all the scores
		r = {}
		for name in colnames:
			r[name] = round(np.mean([ranks[method][name] 
    			for method in ranks.keys()]), 2)
 
		ranks["Mean"] = r
		self.ranks = ranks
		return ranks

	'''
		ranking_name: ['RandomForest', 'Mean', 'Ridge', 'LinReg', 'Lasso', 'RecursiveFeatureElimination','RandomizedLassoStability']
	'''
	def plot_ranking(self, ranking_name='Mean'):
		ranking_df = pd.DataFrame(list(self.ranks[ranking_name].items()), columns= ['Feature',ranking_name])
		ranking_df = ranking_df.sort_values(ranking_name, ascending=False)
		sns.factorplot(x=ranking_name, y="Feature", data = ranking_df, kind="bar", 
               size=20, aspect=.8, palette='coolwarm');
