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

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(68, input_dim=68, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

def apply_threshold(preds):
	p = []
	for pred in preds:
		if pred > 0.5:
			p.append(1)
		elif pred <= 0.5:
			p.append(0)
	return p

# def cross_validation(x, y, unique, index):
#
# 	X_train = x[x.ID != unique[index]]
# 	X_test = x[x.ID == unique[index]]
# 	y_train = y[y.ID != unique[index]]
# 	y_test = y[y.ID == unique[index]]
# 	X_train = X_train.drop(["ID"], axis=1)
# 	X_test = X_test.drop(["ID"], axis=1)
# 	y_train = y_train.drop(["ID"], axis=1)
# 	y_test = y_test.drop(["ID"], axis=1)
#
# 	return X_train, y_train, X_test, y_test

def cross_validation(x, y, unique, index):

	X_train = x[x.ID != unique[index]]
	X_test = x[x.ID == unique[index]]
	y_train = y[y.ID != unique[index]]
	y_test = y[y.ID == unique[index]]
	X_train = X_train.drop(["ID"], axis=1)
	X_test = X_test.drop(["ID"], axis=1)
	y_train = y_train.drop(["ID"], axis=1)
	y_test = y_test.drop(["ID"], axis=1)



	return X_train, y_train, X_test, y_test

def personalized_normalization(subject):
	num_columns = len(x_train.columns)

	# ****************************
	# Normalize the train_x data
	# ****************************
	normalized_train_x = pd.DataFrame()

	for i in range(0, num_columns):
		col_name = x_train.columns[i]

		# Create x, where x the 'scores' column's values as floats
		x_floats = x_train[[col_name]].values.astype(float)

		# Create a minimum and maximum processor object
		min_max_scaler = preprocessing.MinMaxScaler()

		# Create an object to transform the data to fit minmax processor
		x_train_scaled = min_max_scaler.fit_transform(x_floats)

		# Run the normalizer on the dataframe
		normalized_column = pd.DataFrame(x_train_scaled)

		# Append nomalized column to normalized_data dataframe

		normalized_train_x[col_name] = normalized_column.iloc[:, 0]

def normalize(x_train, x_test):
	# *****************
	# * NORMALIZATION *
	# ****************
	# https://chrisalbon.com/python/data_wrangling/pandas_normalize_column/

	num_columns = len(x_train.columns)

	# ****************************
	# Normalize the train_x data
	# ****************************
	normalized_train_x = pd.DataFrame()

	for i in range(0, num_columns):
		col_name = x_train.columns[i]

		# Create x, where x the 'scores' column's values as floats
		x_floats = x_train[[col_name]].values.astype(float)

		# Create a minimum and maximum processor object
		min_max_scaler = preprocessing.MinMaxScaler()

		# Create an object to transform the data to fit minmax processor
		x_train_scaled = min_max_scaler.fit_transform(x_floats)

		# Run the normalizer on the dataframe
		normalized_column = pd.DataFrame(x_train_scaled)

		# Append nomalized column to normalized_data dataframe

		normalized_train_x[col_name] = normalized_column.iloc[:, 0]

	# ***************************
	# Normalize the x_test data
	# ***************************
	normalized_test_x = pd.DataFrame()

	for i in range(0, num_columns):
		col_name = x_test.columns[i]

		# Create x, where x the 'scores' column's values as floats
		x_floats = x_test[[col_name]].values.astype(float)

		# Create a minimum and maximum processor object
		min_max_scaler = preprocessing.MinMaxScaler()

		# Create an object to transform the data to fit minmax processor
		x_test_scaled = min_max_scaler.fit_transform(x_floats)

		# Run the normalizer on the dataframe
		normalized_column = pd.DataFrame(x_test_scaled)

		# Append nomalized column to normalized_data dataframe

		normalized_test_x[col_name] = normalized_column.iloc[:, 0]

	return normalized_train_x, normalized_test_x

def calc_accuracy(predictions, actual):
	# predictions is a 1D python array.
	# actual is a pandas DF of 1 column.
	correct_count = 0
	for i in range(0, len(predictions)):
		# print(f'Prediction: {predictions[i]} vs Actual: {actual.iloc[i][0]}')
		if predictions[i] == actual.iloc[i][0]:
			correct_count += 1

	return correct_count/len(predictions)


def main():
	dataset_path = "Non_uniform_health_alcohol_mgt_feature_final.csv"

	df = pd.read_csv(dataset_path, header=0)

	cols = [0, 69]  # non-feature columns CAN BE DIFF BASED ON FILE
	x = df.drop(df.columns[cols], axis=1)
	y = df[["alcohol_mgt", "ID"]]
	ID = df["ID"]

	# list with all the unique IDs (strings)
	unique = []
	for i in range(0, len(ID)):
		if ID[i] not in unique:
			unique.append(ID.at[i])

	accuracies = []
	#run tests on each fold (one subject left out in each fold).
	for i in range(0, len(unique)):
	# for i in range(0, 1):
		x_train, y_train, x_test, y_test = cross_validation(x, y, unique, i)

		dichotomized_y_train = dichotomize(y_train)
		dich_y_test = dichotomize(y_test)


		# *************
		# * NORMALIZE *
		# *************

		normalized_train_x, normalized_test_x = normalize(x_train, x_test)

		#*************************
		#* Create Baseline model *
		# *************************
		baseline_model = create_baseline()
		baseline_model.fit(normalized_train_x, dichotomized_y_train, epochs=50, batch_size=10, shuffle=True)

		preds = baseline_model.predict(normalized_test_x)
		preds = apply_threshold(preds)

		acc = calc_accuracy(preds, dich_y_test)
		accuracies.append(acc)

	print(f'Average unweighted accuracy accross each fold: {100 * (sum(accuracies)/len(accuracies))}%')



main()
