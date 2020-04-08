import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from keras.callbacks import EarlyStopping
import statistics

data = pd.read_csv("Non_uniform_health_sleep_mgt_feature_final.csv")
# sns.scatterplot(x="ID", y="sleep_mgt", data=data)
# sns.countplot(x = "sleep_mgt", data = data)
# plt.show()

cols = [0, 69]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
y = data[["sleep_mgt", "ID"]]
ID = data["ID"]
predictions = []
unique = []

# unique ids
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

# normalize all data

model = Sequential([Dense(69, input_dim=68, activation='relu'),
                    Dense(1)
                    ])

model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_squared_error'])
for i in range(0, len(unique)):
    X_train = X[X.ID != unique[i]]
    X_test = X[X.ID == unique[i]]
    y_train = y[y.ID != unique[i]]
    y_test = y[y.ID == unique[i]]
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    y_train = y_train.drop(["ID"], axis=1)
    y_test = y_test.drop(["ID"], axis=1)
    # normalizing train features
    num_columns = len(X_train.columns)
    # Im going to do this manually to see if I'm doing something wrong
    for j in range(0, num_columns):
        col_name = X_train.columns[j]
        x_floats = X_train[[col_name]].values.astype(float)
        max_train = max(x_floats)
        min_train = min(x_floats)
        for x in range(0, len(X_train[col_name])):
            if max_train == min_train:
                X_train[col_name].values[x] = 0
            else:
                X_train[col_name].values[x] = ((X_train[col_name].values[x] - min_train) / (max_train - min_train))
        for k in range(0, len(X_test[col_name])):
            if max_train == min_train:
                X_test[col_name].values[k] = 0
            else:
                X_test[col_name].values[k] = (X_test[col_name].values[k] - min_train) / (max_train - min_train)
    print(X_train)
    print(X_test)
    print(y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
    # use 25% of the IDs rather than a 25% of random data
    # split the training data into a validation set
    es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    model.fit(X_train, y_train, batch_size=10, epochs=100, shuffle=True, verbose=1,
              validation_data=(X_val, y_val), callbacks=[es])

    prediction = model.predict(X_test)
    for x in range(0, len(prediction)):
        predictions.append(prediction[x][0])

# get labels in a list
want = y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["sleep_mgt"].at[i])

print(predictions)
print(stats.pearsonr(predictions, labels))
