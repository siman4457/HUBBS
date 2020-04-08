import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv("Non_uniform_health_exercise_mgt_feature_final.csv", header=0)

# sns.countplot(x = "exercise_mgt", data = data)
# plt.show()

cols = [0, 69]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
y = data[["exercise_mgt", "ID"]]
ID = data["ID"]
normalized_X = pd.DataFrame()
num_columns = len(X.columns)
min_max_scaler = preprocessing.MinMaxScaler()
preds = []
unique = []

# get all the unique IDs
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

# normalize the x values and then put back the ID column
for i in range(0, num_columns - 1):
    col_name = X.columns[i]
    x_floats = X[[col_name]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x_floats)
    normalized_values = pd.DataFrame(x_scaled)
    normalized_X[col_name] = normalized_values.iloc[:, 0]
normalized_X["ID"] = ID

# make y binary
# im not normalizing y
for i in range(0, len(y["exercise_mgt"])):
    if int(y["exercise_mgt"][i]) > 0:
        y["exercise_mgt"].at[i] = 1
    else:
        y["exercise_mgt"].at[i] = 0


def unweighted_avg(predictions, labels, threshold):
    duplicate = predictions.copy()
    for values in range(0, len(duplicate)):
        if duplicate[values] > threshold:
            duplicate[values] = 1
        elif duplicate[values] < threshold:
            duplicate[values] = 0
    zero_total = 0
    zero_correct = 0
    one_total = 0
    one_correct = 0
    for x in range(0, len(labels)):
        if labels[x] == 0:
            zero_total += 1
            if duplicate[x] == labels[x]:
                zero_correct += 1
        elif labels[x] == 1:
            one_total += 1
            if duplicate[x] == labels[x]:
                one_correct += 1
    average = (((zero_correct / zero_total) + (one_correct / one_total)) / 2)
    print("The unweighted average with threshold value", threshold, "is", average)
    return average


model = Sequential([Dense(68, input_dim=68, activation='relu'),
                    Dense(1, activation="sigmoid")
                    ])
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
for i in range(0, len(unique)):
    X_train = normalized_X[normalized_X.ID != unique[i]]
    X_test = normalized_X[normalized_X.ID == unique[i]]
    y_train = y[y.ID != unique[i]]
    y_testing = y[y.ID == unique[i]]
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    y_train = y_train.drop(["ID"], axis=1)
    y_test = y_testing.drop(["ID"], axis=1)
    model.fit(X_train, y_train, batch_size=10, epochs=15, shuffle=True, verbose=1)
    predictions = model.predict(X_test)
    # accuracy = model.evaluate(X_test, y_test)
    for i in range(0, len(predictions)):
        preds.append(predictions[i][0])  # list of all of our predictions



# making an array of labels

want = y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["exercise_mgt"].at[i])


unweighted_avg(preds, labels, 0.3)
unweighted_avg(preds,labels, 0.4)
unweighted_avg(preds, labels, 0.5)
unweighted_avg(preds,labels, 0.6)

