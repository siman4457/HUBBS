import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(68, input_dim=68, activation='relu'))
	model.add(Dense(34, activation='relu'))
	model.add(Dense(17, activation='relu'))
	model.add(Dense(1))
	# Compile model

	# opt = SGD(lr=0.01, momentum=0.9)
	# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	model.compile(loss="mean_squared_error", optimizer="adam", metrics=["rmse"]) #we use correlation instead of accuracy for the metrics
	return model

def main():
	dataset_path = "../tiles_data/Non_uniform_health_sleep_mgt_feature_final.csv"

	df = pd.read_csv(dataset_path, header=0)
	x = df.iloc[:, 1:69]
	y = df.iloc[:, 69:70]

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

	# Train Set
	# x = df.iloc[:4000, 1:69]
	# y = df.iloc[:4000, 69:70]
	#
	# # Test Set
	# x_test = df.iloc[4000:, 1:69]
	# y_test = df.iloc[4000:, 69:70]

	min_max_scaler = preprocessing.MinMaxScaler()
	x_normalized = min_max_scaler.fit_transform(x_train)
	x_normalized_test = min_max_scaler.fit_transform(x_test)

	y_normalized = min_max_scaler.fit_transform(y_train)
	y_normalized_test = min_max_scaler.fit_transform(y_test)


	# Create Baseline model
	baseline_model = create_baseline()
	baseline_model.fit(x_normalized, y_normalized, epochs=20, shuffle=True)


	_, accuracy = baseline_model.evaluate(x_normalized_test, y_normalized_test)
	print('Accuracy: %.2f' % (accuracy * 100))

main()