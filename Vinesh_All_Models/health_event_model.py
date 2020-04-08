import keras
import sklearn
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

data = pd.read_csv("Non_uniform_health_event_mgt_feature_final.csv", header=0)

# sns.countplot(x = "tobacco_mgt", data = data)
# plt.show()
# # sns.barplot(x= "ID", y="Cardio_caloriesOut", data = data)
# # There are 154 unique IDs
cols = [0, 69]  # non-feature columns CAN BE DIFF BASED ON FILE
X = data.drop(data.columns[cols], axis=1)
y = data[["event_mgt", "ID"]]
ID = data["ID"]
normalized_X = pd.DataFrame()
num_columns = len(X.columns)
min_max_scaler = preprocessing.MinMaxScaler()
preds = []  # list of predictions
# list with all the unique IDs (strings)

unique = []
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

# Y = data[data["ID"] == unique[0]] to get the rows associated with a particular column
# print(Y)
# training = data[data.ID == unique[0]] # all the values with specific ID


for i in range(0, num_columns - 1):
    col_name = X.columns[i]
    x_floats = X[[col_name]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x_floats)
    normalized_values = pd.DataFrame(x_scaled)
    normalized_X[col_name] = normalized_values.iloc[:, 0]
normalized_X["ID"] = ID
# convert to binary
for i in range(0, len(y["event_mgt"])):
    if int(y["event_mgt"][i]) > 0:
        y["event_mgt"].at[i] = 1
    else:
        y["event_mgt"].at[i] = 0

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
    y_test = y[y.ID == unique[i]]
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    y_train = y_train.drop(["ID"], axis=1)
    y_test = y_test.drop(["ID"], axis=1)
    model.fit(X_train, y_train, batch_size=10, epochs=15, shuffle=True, verbose=1)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    prediction = model.predict(X_test)
    for x in range(0, len(prediction)):
        preds.append(prediction[x][0])

want = y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["event_mgt"].at[i])


unweighted_avg(preds,labels, 0.4)
unweighted_avg(preds, labels, 0.5)
unweighted_avg(preds,labels, 0.6)
unweighted_avg(preds, labels, 0.3)

