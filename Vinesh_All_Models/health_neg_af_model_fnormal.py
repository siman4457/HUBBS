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

data = pd.read_csv("Non_uniform_health_neg_af_mgt_feature_final.csv")
# sns.scatterplot(x="ID", y="sleep_mgt", data=data)
# sns.countplot(x = "sleep_mgt", data = data)
# plt.show()

cols = [0, 69]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
y = data[["neg_af_mgt", "ID"]]
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
    normalized_X = pd.DataFrame()
    normalized_X_test = pd.DataFrame()
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
    for j in range(0, num_columns):
        col_name = X_train.columns[j]
        x_floats = X_train[[col_name]].values.astype(float)
        # print(col_name)
        # print("max:", max(x_floats))
        # print("min:", min(x_floats))
        x_scaled = min_max_scaler.fit_transform(x_floats)
        normalized_values = pd.DataFrame(x_scaled)
        normalized_X[col_name] = normalized_values.iloc[:, 0]
        x_floats_test = X_test[[col_name]].values.astype(float)
        # not using fit here, so it uses the same params
        # source: https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
        x_scaled_train = min_max_scaler.transform(X_test)
        normalized_values_test = pd.DataFrame(x_scaled_train)
        normalized_X_test[col_name] = normalized_values_test.iloc[:, 0]
    X_train, X_val, y_train, y_val = train_test_split(normalized_X, y_train, test_size=0.25, random_state=0)
    # split the training data into a validation set
    es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    model.fit(X_train, y_train, batch_size=10, epochs=100, shuffle=True, verbose=1,
              validation_data=(X_val, y_val), callbacks=[es])

    prediction = model.predict(normalized_X_test)
    for x in range(0, len(prediction)):
        predictions.append(prediction[x][0])

# get labels in a list
want = y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["neg_af_mgt"].at[i])

print(predictions)
print(stats.pearsonr(predictions, labels))
