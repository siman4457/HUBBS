import tensorflow as tf
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy as np




'''
This program trains and reports the accuracy of the alcohol data from TILES
using a FF Neural Network.

References: 

*** https://www.youtube.com/watch?v=T91fsaG2L0s
*** https://www.heatonresearch.com/2017/06/01/hidden-layers.html
'''


def dichotomize(data_outcome):
	dich = []
	try:
		for i in range(0, len(data_outcome)):
			value = float(data_outcome.values[i][0])

			if value > 0.0:
				dich.append(1.0)
			else:
				dich.append(0.0)

	except Exception as e:
		print("Error in dichotomize function:", e)

	data = {'labels': dich}
	return pd.DataFrame(data=data)


# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(68, input_dim=68, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model


def fix_preds(preds):
	p = []
	for pred in preds:
		if pred > 0.5:
			p.append(1)
		elif pred <= 0.5:
			p.append(0)
	return p

def cross_validation(data):
	cols = [0, 69]  # non-feature columns CAN BE DIFF BASED ON FILE
	X = data.drop(data.columns[cols], axis=1)
	y = data[["alcohol_mgt", "ID"]]
	ID = data["ID"]
	# list with all the unique IDs (strings)
	unique = []
	for i in range(0, len(ID)):
		if ID[i] not in unique:
			unique.append(ID.at[i])

	for i in range(0, len(unique)):
		X_train = X[X.ID != unique[i]]
		X_test = X[X.ID == unique[i]]
		y_train = y[y.ID != unique[i]]
		y_test = y[y.ID == unique[i]]
		X_train = X_train.drop(["ID"], axis=1)
		X_test = X_test.drop(["ID"], axis=1)
		y_train = y_train.drop(["ID"], axis=1)
		y_test = y_test.drop(["ID"], axis=1)

def main():
	dataset_path = "Non_uniform_health_alcohol_mgt_feature_final.csv"

	df = pd.read_csv(dataset_path, header=0)
	x = df.iloc[:, 1:69]
	y = df.iloc[:, 69:70]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)

	dichotomized_y_train = dichotomize(y_train)
	dich_y_test = dichotomize(y_test)


	# Generate plot of unnormalized data
	# import matplotlib.pyplot as plt
	# x_train['Cardio_caloriesOut'].plot.bar(title="Cardio_calories_out")
	# plt.show()

	#*****************
	#* NORMALIZATION *
	# ****************
	# https://chrisalbon.com/python/data_wrangling/pandas_normalize_column/

	num_columns = len(x_train.columns)
	normalized_train_x = pd.DataFrame()

	for i in range(0, num_columns):
		col_name =  x_train.columns[i]

		# Create x, where x the 'scores' column's values as floats
		x_floats = x_train[[col_name]].values.astype(float)

		# Create a minimum and maximum processor object
		min_max_scaler = preprocessing.MinMaxScaler()

		# Create an object to transform the data to fit minmax processor
		x_train_scaled = min_max_scaler.fit_transform(x_floats)

		# Run the normalizer on the dataframe
		normalized_column = pd.DataFrame(x_train_scaled)

		#Append nomalized column to normalized_data dataframe

		normalized_train_x[col_name] = normalized_column.iloc[:, 0]

	#Normalize the x_test data
	normalized_test_x = pd.DataFrame()

	for i in range(0, num_columns):
		col_name =  x_test.columns[i]

		# Create x, where x the 'scores' column's values as floats
		x_floats = x_test[[col_name]].values.astype(float)

		# Create a minimum and maximum processor object
		min_max_scaler = preprocessing.MinMaxScaler()

		# Create an object to transform the data to fit minmax processor
		x_test_scaled = min_max_scaler.fit_transform(x_floats)

		# Run the normalizer on the dataframe
		normalized_column = pd.DataFrame(x_test_scaled)

		#Append nomalized column to normalized_data dataframe

		normalized_test_x[col_name] = normalized_column.iloc[:, 0]



	#*************************
	#* Create Baseline model *
	# *************************
	baseline_model = create_baseline()
	baseline_model.fit(normalized_train_x, dichotomized_y_train, epochs=50, batch_size=10, shuffle=True)

	#----------------------------------------------------------------------
	# evaluate baseline model with standardized dataset
	# estimators = []
	# estimators.append(('standardize', StandardScaler()))
	# estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
	# pipeline = Pipeline(estimators)
	# kfold = StratifiedKFold(n_splits=10, shuffle=True)
	# results = cross_val_score(pipeline, x, dichotomized_y_train, cv=kfold)
	# print("Standardized Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
	# ----------------------------------------------------------------------

	preds = baseline_model.predict(normalized_test_x)
	p = fix_preds(preds)
	for z in range(0, len(p)):
		print(f'Prediction: {p[z]} vs Actual: {dich_y_test.iloc[i][0]}')


main()
