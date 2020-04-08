import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy

data = pd.read_csv("Non_uniform_data_Job_interaction_mgt_feature_final.csv", header=0)
# sns.countplot(x="interaction_mgt", data=data)
# plt.show()
# The possible values are 1, 2, 3
# 1 is the greatest, 2 is the least, 3 is the 2nd most
# whats the better way to split?
# split columns and then test and train data


# One hot encoding:
# Transforms categorical labels into vectors of 0s and 1s
# length of there vectors is dependent on the classes we have to classify
# the part of the vector that it corresponds to is a 1 while the remaining are 0s
cols = [0, 70]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
normalized_X = pd.DataFrame()
y = data[["interaction_mgt", "ID"]]
ID = data["ID"]
preds = []
num_columns = len(X.columns)
min_max_scaler = preprocessing.MinMaxScaler()

# get unique columns
unique = []
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])
# normalize x
for i in range(0, num_columns - 1):
    col_name = X.columns[i]
    x_floats = X[[col_name]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x_floats)
    normalized_values = pd.DataFrame(x_scaled)
    normalized_X[col_name] = normalized_values.iloc[:, 0]
normalized_X["ID"] = ID

# decrease all y by 1
for i in range(0, len(y["interaction_mgt"])):
    y["interaction_mgt"].at[i] = y["interaction_mgt"].at[i] - 1


def unweighted_avg(predictions, labels):  # change here too
    duplicate = predictions.copy()
    for values in range(0, len(duplicate)):
        for x in range(0, len(duplicate[values])):
            maximum = max(duplicate[values])
            if duplicate[values][x] == maximum:
                duplicate[values][x] = 1
            elif duplicate[values][x] != maximum:
                duplicate[values][x] = 0
    for values in range(0, len(duplicate)):
        for x in range(0, len(duplicate[values])):
            if duplicate[values][x] == 1:
                duplicate[values] = x
                break
    correct_zero = 0
    total_zero = 0
    correct_one = 0
    total_one = 0
    correct_two = 0
    total_two = 0
    for a in range(0, len(duplicate)):
        if labels[a] == 0:
            total_zero += 1
            if duplicate[a] == labels[a]:
                correct_zero += 1
        elif labels[a] == 1:
            total_one += 1
            if duplicate[a] == labels[a]:
                correct_one += 1
        elif labels[a] == 2:
            total_two += 1
            if duplicate[a] == labels[a]:
                correct_two += 1

    average = ((correct_zero/total_zero) + (correct_one/total_one) + (correct_two/total_two)) / 3
    print("The unweighted accuracy is:", average)
    return average


# one hot encode y
one_hot_y = keras.utils.to_categorical(y["interaction_mgt"], num_classes=3)
# df.loc[df['LastName']=='Smith'].index
# values = normalized_X.loc[normalized_X["ID"] != unique[1]].index
# values2 = normalized_X.loc[normalized_X["ID"] == unique[1]].index
# want1 = one_hot_y[values[0]:values2[0]]

# want2 = one_hot_y[(values2[-1]) + 1: (values[-1] + 1)]
# want = numpy.append(want1, want2, axis=0)
# print(want)
# print(len(want))

model = Sequential([Dense(69, input_dim=69, activation='relu'),
                    Dense(3, activation="softmax")
                    ])

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
for i in range(0, len(unique)):
    X_train = normalized_X[normalized_X.ID != unique[i]]
    X_test = normalized_X[normalized_X.ID == unique[i]]
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    values_not_id_train = normalized_X.loc[normalized_X["ID"] != unique[i]].index # not ID index
    values_id_train = normalized_X.loc[normalized_X["ID"] == unique[i]].index  # use to index where to skip, ID index
    if i == 0:
        y_train = one_hot_y[values_not_id_train[0]:(values_not_id_train[-1] + 1)]
        y_test = one_hot_y[values_id_train[0]: (values_id_train[-1] + 1)]
    else:
        y_train1 = one_hot_y[values_not_id_train[0]:values_id_train[0]]
        y_train2 = one_hot_y[(values_id_train[-1] + 1):(values_not_id_train[-1] + 1)]
        y_train = numpy.append(y_train1, y_train2, axis=0)
        y_test = one_hot_y[values_id_train[0]:(values_id_train[-1] + 1)]
    model.fit(X_train, y_train, batch_size=10, epochs=15, shuffle=True, verbose=1)
    # _, accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    for p in predictions:
        preds.append(p)


# have to index comparison like this: (preds[1][0] == one_hot_y[1][0])
labels = []
want = y.drop(["ID"], axis=1)
for i in range(0, len(want)):
    labels.append(want["interaction_mgt"].at[i])

unweighted_avg(preds,labels)

