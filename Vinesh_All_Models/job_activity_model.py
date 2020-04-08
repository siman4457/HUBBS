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

data = pd.read_csv("Non_uniform_data_Job_activity_mgt_feature_final.csv")
sns.scatterplot(x="ID", y="activity_mgt", data=data)
plt.show()

cols = [0, 69]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
num_columns = len(X.columns)
min_max_scaler = preprocessing.MinMaxScaler()
y = data[["activity_mgt", "ID"]]
ID = data["ID"]
predictions = []
print(X)
print(y)
unique = []
# unique ids
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

"""
model = Sequential([Dense(68, input_dim=68, activation='relu'),
                    Dense(1, activation="sigmoid")
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
    model.fit(X_train, y_train, batch_size=15, epochs=1, shuffle=True, verbose=1)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    prediction = model.predict(X_test)
    print(prediction)
    for x in range(0, len(prediction)):
        predictions.append(prediction[x][0])
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))

want = y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["sleep_mgt"].at[i])

print(labels)
print(predictions)
print(stats.pearsonr(predictions, labels))
"""