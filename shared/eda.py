import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def missing_values(X):
    missing_df = X.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count']>0]
    missing_df = missing_df.sort_values(by='missing_count')

    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()
    return missing_df

def feature_importances(rf_clf, X, figsize=(7,15), plot=True):
    # We will use these feature importances as a baseline later on when we do some feature engineering
    feature_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
    feature_importances.sort_values(ascending=True, inplace=True)
    if plot:
        feature_importances.plot(kind='barh', figsize=figsize)
    return feature_importances