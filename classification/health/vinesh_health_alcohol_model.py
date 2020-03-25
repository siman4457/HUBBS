# selecting certain columns https://www.youtube.com/watch?v=yO9ZihvadpE
#     # week 3
#     # accuracy = accuracy_score(y_test,predictions, normalize=True)
#     # ROC Curve
#     # accuracy score function
#     # correct for 0
#     # correct for 1
#     # do accuracy for different threshhold weighted and unwighted
#     # unweighted is: ((correct 0/total0) + (correct1 / wrong1))/2
#     # try normalizing too
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

data = pd.read_csv("Non_uniform_health_alcohol_mgt_feature_final.csv", header=0)

# sns.countplot(x = "alcohol_mgt", data = data)
# plt.show()
# sns.barplot(x= "ID", y="Cardio_caloriesOut", data = data)
# There are 154 unique IDs
cols = [0, 69]  # non-feature columns CAN BE DIFF BASED ON FILE
X = data.drop(data.columns[cols], axis=1)
y = data[["alcohol_mgt", "ID"]]
ID = data["ID"]
# normalized_X = pd.DataFrame()
# normalized_y = pd.DataFrame()
num_columns = len(data.columns)
preds = [] # list of predictions
# list with all the unique IDs (strings)
unique = []
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

# Y = data[data["ID"] == unique[0]] to get the rows associated with a particular column
# print(Y)
# training = data[data.ID == unique[0]] # all the values with specific ID

# convert to binary
for i in range(0, len(y["alcohol_mgt"])):
    if int(y["alcohol_mgt"][i]) > 0:
        y["alcohol_mgt"][i] = 1



for i in range(0, len(unique)):
    X_train = X[X.ID != unique[i]]
    X_test = X[X.ID == unique[i]]
    y_train = y[y.ID != unique[i]]
    y_test = y[y.ID == unique[i]]
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    y_train = y_train.drop(["ID"], axis=1)
    y_test = y_test.drop(["ID"], axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2, random_state=1)
    model = Sequential([Dense(68, input_dim=68, activation='relu'),
                        Dense(1, activation="sigmoid")
                        ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=10, epochs=10, shuffle=True, verbose=1)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    predictions = model.predict(X_test)
    for i in predictions:
        print(i)
        preds.append(i)

for values in preds:
    print(values)


"""
# normalizing data
for i in range(0, num_columns):
    col_name = X.columns[i]
    x_floats = X[[col_name]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x_floats)
    normalized_values = pd.DataFrame(x_scaled)
    normalized_X[col_name] = normalized_values.iloc[:,0]
"""


# min_max_scaler = preprocessing.MinMaxScaler()
#id_column = data.columns[num_columns - 1]
# for i in data[["ID"]].values:
# print(i)