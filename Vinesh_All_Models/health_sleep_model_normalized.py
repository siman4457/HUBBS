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
import random
from keras import backend
data = pd.read_csv("Non_uniform_health_sleep_mgt_feature_final.csv")
# sns.scatterplot(x="ID", y="sleep_mgt", data=data)
# sns.countplot(x = "sleep_mgt", data = data)
# plt.show()

cols = [0, 69]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
y = data[["sleep_mgt", "ID"]]
# get labels in a list
want = y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["sleep_mgt"].at[i])
ID = data["ID"]
predictions = []
unique = []

# unique ids
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

# normalize all data
model = Sequential([Dense(68, input_dim=68, activation='relu'),
                    Dense(1)
                    ])

model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_squared_error'])
for i in range(0, len(unique)):
    copy_ids = unique.copy()  # using this to get random ids for validation set
    copy_ids.remove(unique[i])
    validation_ids = random.sample(copy_ids, 38)
    normalized_X = pd.DataFrame()
    normalized_X_test = pd.DataFrame()
    final_x_val = pd.DataFrame()
    final_y_val = pd.DataFrame()
    normalized_val = pd.DataFrame()
    X_train = X[X.ID != unique[i]]
    X_test = X[X.ID == unique[i]]
    y_train = y[y.ID != unique[i]]
    y_test = y[y.ID == unique[i]]
    X_val = X
    y_val = y
    for z in validation_ids:
        # this loop is to drop all of the validation ids
        X_train = X_train[X_train.ID != z]
        y_train = y_train[y_train.ID != z]
        X_throw_val = X_val[X_val.ID == z]
        final_x_val = final_x_val.append(X_throw_val, ignore_index=True)  # contains the values of val x ids
        y_throw_val = y_val[y_val.ID == z]
        final_y_val = final_y_val.append(y_throw_val, ignore_index=True)
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    y_train = y_train.drop(["ID"], axis=1)
    y_test = y_test.drop(["ID"], axis=1)
    final_y_val = final_y_val.drop(["ID"], axis=1)
    final_x_val = final_x_val.drop(["ID"], axis=1)

    # normalizing train features
    num_columns = len(X_train.columns)
    for j in range(0, num_columns):
        col_name = X_train.columns[j]
        x_floats = X_train[[col_name]].values.astype(float)
        x_scaled = min_max_scaler.fit_transform(x_floats)
        normalized_values = pd.DataFrame(x_scaled)
        normalized_X[col_name] = normalized_values.iloc[:, 0]
        # finished normalizing training set
        x_floats_test = X_test[[col_name]].values.astype(float)
        # not using fit here, so it uses the same params
        # source: https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
        x_scaled_test = min_max_scaler.transform(x_floats_test)
        # I meant x scaled test, but the values are still right
        normalized_values_test = pd.DataFrame(x_scaled_test)
        normalized_X_test[col_name] = normalized_values_test.iloc[:, 0]
        # finished normalizing the testing set
        x_floats_val = final_x_val[[col_name]].values.astype(float)
        x_scaled_val = min_max_scaler.transform(x_floats_val)
        normalized_values_val = pd.DataFrame(x_scaled_val)
        normalized_val[col_name] = normalized_values_val.iloc[:, 0]
        # finished normalizing the validation set

    es = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    model.fit(normalized_X, y_train, batch_size=10, epochs=100, shuffle=True, verbose=1,
              validation_data=(normalized_val, final_y_val), callbacks=[es])

    prediction = model.predict(normalized_X_test)
    for x in range(0, len(prediction)):
        predictions.append(prediction[x][0])
    backend.clear_session()

print(predictions)
print(labels)
print(stats.pearsonr(predictions, labels))
