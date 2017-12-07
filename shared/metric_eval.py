import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn as sk

def classification_report(y_true,y_pred,
						classes=None,
						title='Confusion matrix',
						cmap=plt.cm.Blues):
	
	cm = sk.metrics.confusion_matrix(y_true, y_pred)
	print(sk.metrics.classification_report(y_true,y_pred))
	print('Accuracy score', str(round(sk.metrics.accuracy_score(y_true,y_pred), 2) * 100) + '%', )
	plt.imshow(cm, cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).round(2)
		# Max value is 100, make threshold half that

	# Add labels to the picture
	thresh = cm.max()/2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, str(cm[i,j]) + "\n\n" + str(cm_norm[i, j]) + "%",
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.grid(b=False)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return cm